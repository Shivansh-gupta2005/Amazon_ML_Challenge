# Business Entity Resolution — Methodology

> Fill-in of the challenge's `Documentation_template.md`. It describes the
> methodology, the blocking/candidate-generation strategy, the model architecture
> and feature engineering, and other relevant details (evaluation, licensing,
> fair-play, reproducibility).

## 1. Problem framing

We are given business records from three independent sources. **Source 1 (S1)** is
the deduplicated reference; for every S1 record we must output the set of matching
records from **Source 2 (S2)** and **Source 3 (S3)** — possibly empty (a singleton),
one, or many. There are no shared identifiers, and the fields (`business_name`,
`business_address`, `country`) are noisy and inconsistent.

We treat this as a **supervised pairwise matching** problem wrapped in a standard
entity-resolution pipeline:

```
normalise → block (generate candidates) → featurise pairs → score (LightGBM)
          → threshold + conflict resolution → per-S1 match lists
```

The scoring metric is **macro-averaged F0.5** over S1 entities. Because F0.5 weights
precision twice as heavily as recall, a false merge (linking two different
businesses) costs roughly twice a missed link. Every design decision below leans
toward **precision**, while blocking protects recall so the model has the true
matches available to choose from.

## 2. Methodology (overview)

1. **Normalisation** of names and addresses to absorb the documented noise
   (abbreviations, legal-suffix inconsistencies, `&`/`and`, punctuation,
   transliteration/accents, address abbreviations, landmarks, missing components).
2. **Blocking / candidate generation** that pairs each S1 record with a small set of
   plausible S2/S3 records, drastically reducing the O(N²) comparison space while
   keeping recall high. This set is written verbatim to `candidate_pairs.tsv`.
3. **Feature engineering** on each candidate pair: string-similarity signals for
   name and address, postal/number agreement, country agreement, and a source flag.
4. **A LightGBM binary classifier** that scores each pair with P(match).
5. **A decision layer**: a threshold tuned for macro F0.5, plus a target-uniqueness
   constraint that reflects S1 being a deduplicated reference.
6. **Validation**: a held-out split of S1 entities scored with the exact F0.5 formula,
   used to tune the threshold and estimate leaderboard performance.

The pipeline is fully deterministic (fixed seeds) and reproducible from the provided
data with only the pinned open-source dependencies.

## 3. Candidate generation / blocking strategy

Blocking sets the **recall ceiling** for the whole system, so it is deliberately
generous but still selective. We union two complementary blockers, both scoped to
records sharing the same `country` label (a natural, data-driven partition — never a
hard-coded country list):

1. **TF-IDF character n-gram cosine top-n (primary).**
   A `char_wb` TF-IDF vectoriser (n-grams 4–5) is fit over the business names of all
   records. For each S1 record we retrieve its **top-k** (default k = 20) nearest S2
   and S3 records by cosine similarity using `sparse_dot_topn`, which keeps only the
   top-k entries per row and never densifies the product. Character n-grams are robust
   to typos, spacing, punctuation and suffix changes, which makes this a strong recall
   driver.

   **The key to scaling** this to tens of millions of records is dropping n-grams with
   a high **absolute** document frequency (`name_max_df`). Common substrings (which
   would otherwise make the sparse product effectively all-pairs — quadratic) are
   removed, leaving only rare, discriminative n-grams; the product then touches very
   few records per query. An absolute cap (rather than a fraction) keeps this behaviour
   identical whether the corpus is 10³ or 10⁷ records. Blocking is country-scoped and
   Source-1 is streamed in chunks so memory stays bounded. On adversarially noisy
   synthetic data this reaches ~0.95 pair recall at ~5k Source-1 records/second/core.

2. **Address char n-gram cosine top-n (co-primary).**
   The same sparse top-n approach applied to the normalised *address*. This is the key
   recall driver on real data: a true match very often shares the address but has a
   completely different name — a rebrand, a domain/handle form (`acme.com`, `@acme`),
   or heavy typos — which name-only blocking can never surface. Unioned with the name
   blocker, it lifts pair-recall substantially (on adversarial synthetic data: 0.94 →
   0.99).

3. **Postal-code blocking (recall booster).**
   Records that share an exact postal code (5–6 digit US ZIP / India PIN / France code)
   are unioned in, capped per code to avoid large blocks. Postal is treated as an
   address signal, so it is not gated on name similarity.

The union (de-duplicated on the pair) is the candidate set fed to the model, and is
exactly what we write to `candidate_pairs.tsv` — so every predicted match is
guaranteed to have appeared as a candidate.

**Why the shared TF-IDF space matters:** blocking and the cosine *feature* use the
same vectoriser, so the similarity the model sees is consistent with how candidates
were selected.

**Diagnostics.** The pipeline reports pair-recall, entity-full-recall (fraction of
non-singleton S1 entities whose *entire* true match set is captured) and average
candidates per S1 on the validation split, so blocking can be tuned against its
recall ceiling and reduction ratio.

## 4. Model architecture and feature engineering

### 4.1 Model

- **LightGBM gradient-boosted decision trees** (`LGBMClassifier`, binary objective).
  MIT-licensed and a few hundred shallow trees — orders of magnitude below the
  8B-parameter limit. GBDTs handle heterogeneous, differently-scaled similarity
  features, non-linear interactions (e.g. "high name similarity **and** matching
  postal") and missing-signal indicators without feature scaling.
- Key settings: `learning_rate = 0.05`, `num_leaves = 63`, subsample and
  column-subsample 0.8, L2 regularisation, with **early stopping** on a held-out
  slice using **binary log-loss** (log-loss keeps improving as probabilities sharpen,
  so it does not halt the moment the classes separate). The final model is refit on
  all labelled data at the chosen number of trees, floored so it can never collapse
  to a degenerate few-tree model.
- **Class imbalance** (matches are rare among candidate pairs) is handled by the
  **decision threshold**, not by oversampling — this keeps probabilities meaningful
  and gives direct control over the precision/recall trade-off that F0.5 cares about.
- **Training-time positive injection:** true-positive pairs that blocking happened to
  miss are injected into the *training* set only (never into the candidate output), so
  the classifier still learns what a positive looks like. This does not affect
  inference or the reported blocking recall.

### 4.2 Normalisation (feature substrate)

- **Names:** ASCII-fold (accents/transliteration) → lowercase → `&`→`and` →
  expand abbreviations (Corp→Corporation, Pvt→Private, Ltd→Limited, Intl→International,
  …) → a **core name** with legal suffixes and connectors removed. We keep both the
  full normalised name and the core name.
- **Addresses:** ASCII-fold → expand postal abbreviations (Rd→Road, St→Street,
  Ave→Avenue, Opp→Opposite, Nr→Near, Ngr→Nagar, directions, …) → extract the
  postal code and all numeric tokens (street/block numbers).

### 4.3 Pairwise features (15, streamed in bounded batches)

Features are computed one batch of candidate pairs at a time (cheap features
vectorised with numpy/scipy; only the fuzzy ratios use a per-pair rapidfuzz call), so
memory is flat regardless of the total candidate count. The address cosine is read
from a precomputed address matrix, and the name cosine is carried over from blocking,
so no re-vectorising happens during featurisation. The families are:

**Name similarity**
- TF-IDF char n-gram cosine (shared with blocking)
- Token Jaccard and containment on core-name tokens
- rapidfuzz `token_sort_ratio`, `token_set_ratio`, `partial_ratio`, `ratio`
- First-token match, acronym match (initials)
- Character-length ratio, token-count difference

**Address similarity**
- TF-IDF char n-gram cosine
- Token Jaccard and containment
- rapidfuzz `token_set_ratio`, `partial_ratio`
- Address-present flag, postal-present flag, **postal exact match**
- Street-number Jaccard and any-number-match flag

**Context**
- Country match
- Target-source flag (S2 vs S3), since the two sources can behave differently

Missingness is encoded explicitly (present/absent flags) rather than imputed, so the
model can learn how much to trust a signal when a component is missing — a common
situation given partial addresses and dropped PIN codes.

On our validation runs the most informative features are a mix of **name** signals
(`rf_token_set`, `rf_token_sort`, name cosine) and **address** signals
(numeric/street-number overlap, address Jaccard/cosine, postal match) — confirming
that discriminating true matches from same-name-different-place look-alikes relies on
address as much as name.

## 5. Decision layer / conflict resolution

- **Threshold.** Scanned on the validation split to maximise macro F0.5, with a light
  floor. Because F0.5 is precision-heavy, the objective itself pushes the threshold up.
- **Target uniqueness.** S1 is a deduplicated reference and each S2/S3 record refers
  to a single real business, so a given S2/S3 record is awarded to **at most one** S1
  entity — the highest-scoring one above threshold. This removes a whole class of
  false merges. (An S1 entity may still match *many* S2/S3 records — we never cap the
  S1 side.)
- **Singletons.** An S1 entity with no candidate above threshold gets an empty match
  list, which scores a full 1.0 when it is truly a singleton.

## 6. Validation and evaluation

- We split S1 **entities** (not pairs) into train/validation, so no candidate pair
  leaks across the split.
- Scoring implements the challenge formula exactly, including the singleton rules
  (true-empty & pred-empty → 1.0; any prediction on a true singleton → 0.0), and the
  macro average over all S1 entities in the evaluation set.
- The same held-out split is used to choose the decision threshold; the final model
  is then refit on all labelled data for test inference.

**Reproducible smoke test.** A synthetic-data generator reproduces the documented
noise patterns (including same-name/different-address hard negatives) and adds
**France** as an unseen test country. At ~150k test entities the pipeline reaches
~0.97 macro F0.5 with ~0.95 blocking recall, **1.00 precision-on-overlap**, correct
singletons, and the unseen country handled — passing the format validator end to end.
(Real F0.5 will differ; synthetic names are cleaner than reality.)

## Scale and complexity

The pipeline is built to run on the full challenge data (~2.2M Source-1 and ~5M each
Source-2/3, ~12.5M records per split) on a single machine:

- **Blocking is sub-quadratic.** High-document-frequency n-grams are pruned, so the
  sparse cosine top-n only compares records sharing rare n-grams — near-linear in
  practice rather than the N² of an all-pairs cosine.
- **Training uses a subsample** of Source-1 entities (default 150k); a sample trains
  an equally good pairwise matcher, and its features are computed once and reused for
  the final refit. Inference still covers every Source-1 record.
- **Featurisation and scoring stream** in bounded batches; both output TSVs are written
  by a single sorted sweep, so nothing proportional to the 10⁸-scale candidate set is
  held in memory.
- **Vectorised normalisation** (whole-column pandas string ops) replaces per-row Python.

Expected end-to-end time on the full data is roughly 45–90 minutes on a multi-core
laptop with 16 GB+ RAM, tunable via `--max-train-entities`, `--knn`, `--batch-size`
and `--max-features`.

## 7. Open-set country handling

`country` is treated as an opaque string label everywhere: it is only a blocking key
and a boolean match feature. Nothing is hard-coded, filtered, or one-hot-encoded to
`{US, India}`, so a third label (France) in the test set flows through the pipeline
unchanged and every test entity — France included — appears in the submission.

## 8. Fair play, licensing, and reproducibility

- **No external data or lookups.** Every signal is computed from the provided files.
  No commercial ER services, no government registries, no geocoding APIs, no internet
  augmentation.
- **Model license & size.** LightGBM (MIT) is the final model, well under the 8B
  parameter limit. All other dependencies are BSD/MIT (numpy, pandas, scipy,
  scikit-learn, rapidfuzz).
- **Determinism.** Fixed random seeds; the two output TSVs regenerate from the
  training/test data using only the contents of `code/business_entity_resolution/`.

## 9. Output files

- `output/matching_results.tsv` — final matches, one row per S1 entity
  (`source1_entity_id`, `matched_entity_ids`). This is the leaderboard upload.
- `output/candidate_pairs.tsv` — the exact candidate set scored by the model, one row
  per S1 entity (`source1_entity_id`, `candidate_entity_ids`).

Both are tab-separated, one row per S1 entity, empty list for no matches, S2-/S3- IDs
only, no duplicates within a list, and every matched ID is a subset of that entity's
candidates — verified by `utils/validate_submission.py`.

## 10. How to reproduce

See `code/business_entity_resolution/README.md` for exact commands. In short, from
`code/business_entity_resolution/`:

```bash
pip install -r requirements.txt
python src/run.py --data-dir /path/to/dataset --output-dir ../../output --model-dir artifacts
python3 utils/validate_submission.py --matching ../../output/matching_results.tsv \
        --candidate ../../output/candidate_pairs.tsv --test-dir /path/to/dataset/test
```

## 11. Possible extensions

- Per-country or per-target-source thresholds / models.
- Phonetic and sorted-neighbourhood blocking keys for additional recall on heavy
  transliteration.
- Transitive consistency checks across S2↔S3 co-references.
- Probability calibration (isotonic) for a more interpretable threshold.
