# Business Entity Resolution — Pipeline

Links records across three independent business data sources. **Source 1** is the
deduplicated reference; for every Source 1 record the pipeline finds its matching
records in **Source 2** and **Source 3** (zero, one, or many), and writes the two
required TSVs. It is tuned for the challenge's precision-heavy **F0.5** metric and
scales to tens of millions of records on a single machine.

```
blocking (name + address + squished-name cosine top-n, + postal)
      ->  pairwise features  ->  LightGBM matcher
      ->  per-country threshold (cold-calibrated for unseen countries)
          + target-uniqueness resolution  ->  matching_results.tsv
```

The final model is **LightGBM** (MIT license, a few hundred trees — far below the
8B-parameter cap). No external data, APIs, or lookups are used.

**Unseen countries (France).** The test set includes France, which never appears in
training, so a single threshold tuned on the (US+India) validation split is calibrated
for the wrong distribution. The pipeline estimates the correct decision bar for an
unseen country by **leave-one-country-out (LOCO)** calibration — it drops each training
country in turn, treats it as unseen, tunes a threshold on it, and applies the median
of those thresholds to any test country not seen in training. Seen countries keep the
in-distribution threshold. This is the main v4 change and directly targets the
validation→leaderboard gap that France causes.

---

## 1. Install

```bash
cd code/business_entity_resolution
python -m pip install -r requirements.txt
```

Use `python -m pip` with the **same** interpreter you will run the script with
(on Windows, `py -3.12 -m pip ...` then `py -3.12 src/run.py ...`). Python 3.11/3.12
recommended. Dependencies: numpy, pandas, scipy, scikit-learn, lightgbm, rapidfuzz,
sparse_dot_topn (all pinned, all permissively licensed).

## 2. Data layout

```
dataset/
  train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
  test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

## 3. Run end-to-end (data → blocking → matching → output)

From `code/business_entity_resolution/` (dataset/ and output/ sit two levels up in
the standard `student_resource/` layout):

```bash
python src/run.py --data-dir ../../dataset --output-dir ../../output --model-dir ../../artifacts
```

This trains + validates on the training data (reporting macro **F0.5**), refits the
final model, then predicts on the test set and writes:
- `../../output/matching_results.tsv` — final matches (upload this to the leaderboard)
- `../../output/candidate_pairs.tsv` — the candidate set fed to the model

### Other modes

```bash
python src/run.py --mode train    --data-dir ../../dataset   # train + validate only
python src/run.py --mode predict  --data-dir ../../dataset   # load saved model, predict test
```

### Performance / scale flags

| flag | meaning | default |
|------|---------|---------|
| `--max-train-entities N` | train the matcher on a sample of N Source-1 entities (0 = all) | 150000 |
| `--knn N` | candidates kept per target source in blocking (lower = faster, less recall) | 20 |
| `--batch-size N` | candidate pairs featurised/scored per batch (lower = less RAM) | 500000 |
| `--max-features N` | TF-IDF vocabulary cap | 400000 |
| `--min-threshold X` | floor for the tuned decision threshold | 0.35 |
| `--no-postal` | disable postal-code blocking | off |
| `--no-cold-country` | disable LOCO calibration; unseen countries use the normal threshold | off |
| `--cold-threshold X` | force the threshold for unseen countries (skips LOCO) | (auto via LOCO) |
| `--corroboration-delta X` | rescue borderline pairs within X of threshold that share a postal code or have high address cosine (0 = off) | 0 |

## 4. Expected run time and memory (real challenge scale)

On the full challenge data (~2.2M Source-1 and ~5M each Source-2/3, ~12.5M records
per split) expect roughly **45–90 minutes end-to-end on a multi-core laptop**, and
**16 GB+ RAM is recommended** (32 GB comfortable). The pipeline prints a timestamped
progress line for every stage. If you are RAM-constrained, lower `--batch-size`,
`--max-features`, and `--knn`; to iterate faster, lower `--max-train-entities`.

Why it scales: blocking never compares all pairs. The TF-IDF vectoriser drops
high-document-frequency character n-grams (an **absolute** cap, `name_max_df`), so the
sparse top-n product (`sparse_dot_topn`) only touches rare, discriminative n-grams;
normalisation is vectorised; and features + scoring are streamed in bounded batches.

## 5. Validate the output before submitting

```bash
python3 utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```

Prints `PASS` (exit 0) or a numbered list of problems (exit 1). Stdlib only. If the
challenge ships its own `utils/validate_submission.py`, that copy is authoritative —
this one mirrors the same rules.

## 6. Reproduce the smoke test (no real data needed)

A synthetic generator reproduces the documented noise (name abbreviations, `&`/`and`,
typos, transposition; address abbreviations, dropped components, landmarks) plus
same-name/different-address hard negatives and France as an unseen country.

```bash
python3 utils/make_synthetic_data.py --out-dir dataset --seed 42 --n-train 100000 --n-test 100000
python src/run.py --data-dir dataset --output-dir output --model-dir artifacts
python3 utils/validate_submission.py --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

On synthetic data the pipeline reaches ~0.97 macro F0.5 with ~0.95 blocking recall
and 100% precision-on-overlap, passing validation and handling the unseen country and
singletons. (Real F0.5 will differ; synthetic names are cleaner than reality.)

## 7. v5 — semantic embeddings (optional, GPU)

Character n-grams match on shared letters; they miss matches that share *meaning* but
few characters — the France gap. The v5 stage adds a **local multilingual sentence
embedding** as (a) a fourth blocker (FAISS ANN nearest-neighbours) and (b) an
`embed_cos` feature, plus an optional cross-encoder reranker for borderline pairs. It is
**off by default** — the CPU pipeline is byte-for-byte unchanged — and turns on with
`--use-embeddings`.

**Compliance.** The model (`paraphrase-multilingual-MiniLM-L12-v2`, Apache-2.0, ~118M
params — far under the 8B cap) is a *pretrained backbone* downloaded once and run
**entirely offline**. It is never queried as an external service and holds no knowledge
of the dataset's businesses, so it is not external-data augmentation. No external
databases, APIs, or internet lookups about the entities are used.

**Setup (on a GPU box — an AWS Deep Learning AMI already has CUDA + PyTorch):**
```bash
pip install -r requirements-embeddings.txt
```
**Run with embeddings:**
```bash
python src/run.py --data-dir ../../dataset --output-dir ../../output --model-dir ../../artifacts \
    --use-embeddings                       # add --use-cross-encoder to also rerank borderline pairs
```
The embedding matrix is encoded once (the GPU-heavy step) and cached to a float32
**memmap** on disk (~19 GB at full scale — so a memmap, not RAM), then reused by the
blocker and the feature. FAISS search runs on CPU per country-source group.

**Hardware.** Encoding ~12.5M records wants a GPU (g5.xlarge / A10G ≈ 1–1.5 h;
CPU is impractical). For headroom on the memmap + FAISS, **g5.2xlarge (32 GB RAM)** is
the safe choice. Stop the instance when idle — GPU boxes bill per second.

**Offline test (no model download, no GPU).** The encoder and reranker are pluggable, so
the whole integration can be exercised on CPU with deterministic stand-ins:
```bash
python src/run.py --data-dir dataset --output-dir output --model-dir artifacts \
    --use-embeddings --embed-backend hashing \
    --use-cross-encoder --cross-encoder-backend stub
```

| flag | meaning | default |
|------|---------|---------|
| `--use-embeddings` | enable the embedding blocker + `embed_cos` feature | off |
| `--embed-backend` | `sentence-transformers` (real) or `hashing` (offline test) | sentence-transformers |
| `--embed-model` | sentence-transformers model id | paraphrase-multilingual-MiniLM-L12-v2 |
| `--embed-device` | `cuda` / `cpu` | auto |
| `--knn-embed` | embedding neighbours per source in blocking | 15 |
| `--embed-min-cosine` | embedding cosine floor for blocking | 0.55 |
| `--use-cross-encoder` | rerank borderline pairs with a cross-encoder (experimental) | off |

## 8. v6 — fine-tuned cross-encoder (GPU)

The data diagnostic (`utils/diagnose_data.py`) shows there is **no exact key** to exploit
and that matching on exact name has only **~0.41 precision** — many *different* businesses
share a name, and the **address** is what decides identity. The strongest remaining lever
is a cross-encoder **fine-tuned on the provided labels** to learn exactly that
disambiguation. This is what most likely separates the top solutions.

**Compliance.** Trains only on the provided `train_ground_truth`; the base model
(`mmarco-mMiniLMv2-L12-H384`, Apache-2.0, ~118M params) is well under the 8B cap and runs
locally. No external data.

**Step 1 — fine-tune** (GPU box; `pip install -r requirements-embeddings.txt` first):
```bash
python3 utils/train_cross_encoder.py --data-dir "<dataset>" --out-dir "$HOME/ce_model"
```
It reuses the pipeline's normalisation, index and blocking to mine **positives** (ground-
truth pairs plus the true pairs blocking missed) and **hard negatives** (the highest
name-similarity NON-matches — same name, different address), composes each record as
`name | address | country`, and fine-tunes the cross-encoder. ~30–60 min on an A10G.
Test the pair mining offline with `--dry-run` (no GPU, no download).

**Step 2 — predict with it** (point the reranker at the fine-tuned model, widen the band):
```bash
python3 src/run.py --data-dir "<dataset>" --output-dir "$HOME/out_v6" --model-dir "$HOME/art_v6" \
    --use-embeddings \
    --use-cross-encoder --cross-encoder-model "$HOME/ce_model" --cross-encoder-band 0.30
```
The fine-tuned model re-scores every pair whose LightGBM score is within the band of the
threshold — flipping same-name/different-address false merges out and pulling true matches
in. With a fine-tuned model, a heavier blend is reasonable (`Config.cross_encoder_weight`,
try 0.6–0.8).

| training flag | meaning | default |
|------|---------|---------|
| `--out-dir` | where the fine-tuned model is saved | (required) |
| `--base-model` | base cross-encoder to fine-tune | mmarco-mMiniLMv2-L12-H384 |
| `--train-entities` | Source-1 entities sampled to mine pairs | 120000 |
| `--epochs` | fine-tuning epochs | 2 |
| `--dry-run` | build + report pairs only, no training (offline) | off |

---

## Project layout

```
src/
  run.py                 # CLI entry point
  ber/
    config.py            # all hyper-parameters + paths
    io_utils.py          # robust TSV read/write, ground-truth parsing
    normalize.py         # vectorised name/address normalisation
    vectorize.py         # char n-gram TF-IDF index + sparse top-n + batch cosine
    blocking.py          # candidate generation (rare-n-gram cosine top-n + postal)
    features.py          # streamed pairwise feature engineering
    embed.py             # v5: multilingual embeddings + FAISS ANN blocker
    rerank.py            # v5: optional cross-encoder reranker (borderline pairs)
    model.py             # LightGBM train / predict / save / load
    resolve.py           # threshold + target-uniqueness decision layer
    evaluate.py          # F0.5 (macro over Source1) + threshold tuning
    pipeline.py          # orchestration (context, train / validate / predict)
utils/
  validate_submission.py # format validator (stdlib only)
  make_synthetic_data.py # synthetic data for the smoke test
  diagnose_data.py       # exact-key / match-structure diagnostic on labelled data
  train_cross_encoder.py # v6: fine-tune a cross-encoder on the provided labels
requirements.txt
README.md
```

## Design notes (precision-first, for F0.5)

- The decision threshold is tuned directly on a held-out split to maximise macro F0.5,
  and a **target-uniqueness** rule gives each Source-2/3 record to at most one Source-1
  entity (its best-scoring one), since Source 1 is deduplicated.
- **Country is an open set** — used as a blocking key, a feature, and a routing key for
  the decision threshold, never hard-coded — so an unseen label like France works
  unchanged and gets its own LOCO-calibrated bar.
- **Cold-country calibration (v4)** measures the right threshold for an unseen country
  instead of assuming it: the LOCO folds are logged (e.g. `us: t=0.36`, `india: t=0.91`)
  so you can see the direction the shift pushes. Per-row routing means changing the cold
  bar affects *only* unseen-country rows; seen countries are bit-for-bit unchanged.
- **Two v4 features** — `postal_prefix_match` (region-level postal agreement, robust to
  the last digits being noisy) and `shared_long_token` (a shared distinctive ≥5-char name
  token that survives suffix noise and rebrands) — both cheap and language-agnostic.
- `candidate_pairs.tsv` is exactly what the model scores, so every matched ID appears
  there. Determinism via fixed seeds (`config.RANDOM_SEED`).
