"""Configuration: paths and hyper-parameters for the entity-resolution pipeline.

Tuned to scale to tens of millions of records on a single machine:
  * blocking uses sparse top-n matrix products (no dense blow-up)
  * normalisation is vectorised (pandas string ops, no per-row Python)
  * features + scoring are streamed in bounded batches
  * the matcher is trained on a subsample of Source-1 entities
`run.py` overrides any of these from the command line.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

RANDOM_SEED = 42


@dataclass
class Config:
    # ------------------------------------------------------------------ paths
    data_dir: str = "dataset"
    output_dir: str = "output"
    model_dir: str = "artifacts"

    train_subdir: str = "train"
    test_subdir: str = "test"

    source1_name: str = "source1.tsv"
    source2_name: str = "source2.tsv"
    source3_name: str = "source3.tsv"
    ground_truth_name: str = "ground_truth.tsv"
    train_prefix: str = "train_"
    test_prefix: str = "test_"

    # --------------------------------------------------------------- blocking
    # char n-gram TF-IDF for blocking + name cosine. The critical knob is name_max_df:
    # dropping n-grams that occur in a large fraction of records removes the common
    # n-grams that make an all-pairs sparse product quadratic, so blocking stays fast
    # while still ranking by cosine (which is typo-robust, unlike exact-token blocking).
    name_ngram_range: Tuple[int, int] = (4, 5)
    addr_ngram_range: Tuple[int, int] = (3, 4)
    # squished name = the name with spaces removed, matched with a plain char n-gram
    # (not char_wb). This catches domain/handle forms like "acme.com" / "@acme" that
    # collapse to one token and share no per-word n-grams with the spaced name.
    squish_ngram_range: Tuple[int, int] = (4, 5)
    # ABSOLUTE document-frequency caps (ints), so behaviour is scale-invariant:
    # a fraction would over-prune tiny corpora and, on tens of millions of rows,
    # keep very common n-grams that make blocking quadratic. Capping the absolute df
    # bounds the blocking cost (~sum of df^2) regardless of dataset size.
    # Absolute df caps: moderate so blocking stays fast at 10M+ scale (cost ~ df^2),
    # but high enough to keep recall. Three blockers (name+addr+squish) share the work,
    # so each can be capped tighter than a single blocker would need.
    name_max_df: int = 12000
    addr_max_df: int = 12000
    squish_max_df: int = 12000
    max_features_name: int = 400_000     # vocab cap -> bounded memory at scale
    max_features_addr: int = 400_000
    min_df: int = 2                      # drop hapax n-grams (noise + memory)

    knn_name: int = 25                   # name-cosine candidates per Source-1 per target source
    knn_min_cosine: float = 0.06
    # ADDRESS blocking: many true matches share an address but have wildly different
    # names (rebrands, domain/handle forms, heavy typos), so we also pair records by
    # address similarity, independent of name -- this is the main recall driver on
    # real data. Unioned with name blocking.
    use_addr_blocking: bool = True
    knn_addr: int = 12                   # address-cosine candidates per Source-1 per target source
    addr_min_cosine: float = 0.30        # addresses are long -> a higher floor is safe
    # SQUISHED-NAME blocking: catches domain/handle/one-token name forms.
    use_squish_blocking: bool = True
    knn_squish: int = 12
    squish_min_cosine: float = 0.30
    use_postal_blocking: bool = True
    postal_min_name_cosine: float = 0.0  # postal is an address signal; don't gate on name
    postal_max_per_key: int = 40
    block_within_country: bool = True
    knn_chunk_rows: int = 20_000         # Source-1 rows per blocking chunk (bounded memory)
    sdt_threads: Optional[int] = None    # sparse_dot_topn threads (None = all cores)

    # --------------------------------------------------------------- batching
    feature_batch_size: int = 500_000    # candidate pairs featurised/scored per batch

    # ------------------------------------------------------------------ model
    # subsample Source-1 entities for TRAINING only (a sample trains an equally
    # good pairwise matcher); None = use all. Inference always covers every entity.
    max_train_entities: Optional[int] = 150_000

    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "boosting_type": "gbdt",
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_jobs": -1,
        "random_state": RANDOM_SEED,
        "verbose": -1,
    })
    early_stopping_rounds: int = 50
    min_final_estimators: int = 150
    inject_missed_positives: bool = True

    # ---------------------------------------------------------- decision layer
    min_threshold: float = 0.35
    threshold_grid_steps: int = 91
    enforce_target_uniqueness: bool = True

    # COLD-COUNTRY calibration (the v4 fix for the unseen-country gap).
    # The test set contains France, which never appears in training, so a threshold
    # tuned on the (US+India) validation split is calibrated for the wrong
    # distribution. We estimate the right bar for an unseen country by leave-one-
    # country-out (LOCO): drop one training country, treat it as "unseen", and tune a
    # threshold on it. The aggregate of those LOCO thresholds is applied to any test
    # country not seen in training (France). Seen countries keep the normal threshold.
    use_cold_country_threshold: bool = True
    cold_threshold_override: Optional[float] = None  # set to skip LOCO and force a value
    cold_min_positive_eval: int = 100    # skip a LOCO fold if the held-out country has too few positives
    cold_threshold_floor: float = 0.20   # never let the cold bar collapse below this
    cold_threshold_cap: float = 0.95
    cold_fold_estimators: int = 200      # trees per (fast) LOCO fold model

    # Optional corroboration tier (default OFF: delta=0 makes it a no-op). When >0, a
    # pair scoring within `corroboration_delta` below its threshold is still accepted
    # if it has a strong address signal (same postal code, or address cosine high).
    # Recovers recall on borderline pairs without lowering the global bar.
    corroboration_delta: float = 0.0
    corroboration_addr_cos: float = 0.75

    # ------------------------------------------------------------- embeddings (v5)
    # Local multilingual sentence embeddings: a semantic blocker + an `embed_cos`
    # feature. OFF by default so the CPU pipeline is byte-for-byte unchanged; turn on
    # with --use-embeddings (best on a GPU box). See ber/embed.py for the compliance note.
    use_embeddings: bool = False
    embed_backend: str = "sentence-transformers"   # or "hashing" (offline test/CPU stand-in)
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embed_device: Optional[str] = None             # None -> cuda if available, else cpu
    embed_text_col: str = "name_key"               # which normalised column to embed
    embed_batch_size: int = 100_000                # rows per memmap write chunk
    embed_enc_batch: int = 256                     # model.encode() batch size
    embed_hash_dim: int = 256                      # HashingEncoder dim (test backend)
    knn_embed: int = 15                            # embedding neighbours per Source-1 per source
    embed_min_cosine: float = 0.55                 # multilingual embeddings -> a fairly high floor
    embed_hnsw_max_exact: int = 200_000            # gallery size above which HNSW replaces exact search

    # Optional cross-encoder reranker (v5, default OFF). Re-scores borderline pairs
    # (LightGBM score within `cross_encoder_band` of the threshold) with a multilingual
    # cross-encoder and blends the two scores. Heavier; needs a GPU; experimental.
    use_cross_encoder: bool = False
    cross_encoder_backend: str = "sentence-transformers"   # or "stub" (offline test)
    # cross_encoder_model may be a HuggingFace id OR a local path to a v6 fine-tuned model
    cross_encoder_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    cross_encoder_band: float = 0.15
    cross_encoder_weight: float = 0.5
    cross_encoder_enc_batch: int = 128

    # ---------------------------------------------- v6: cross-encoder fine-tuning
    # Train a cross-encoder ON THE PROVIDED LABELS to crack the same-name/different-
    # branch trap (exact-name precision is only ~0.41 -- the address decides it). Hard
    # negatives (high name-cosine NON-matches) are the core teaching signal. Compliant:
    # trains only on the provided ground truth; base model is Apache-2.0 and <8B params.
    ce_base_model: str = "nreimers/mmarco-mMiniLMv2-L12-H384-v1"   # multilingual, Apache-2.0
    ce_train_entities: int = 120_000     # Source-1 entities sampled to build training pairs
    ce_neg_ratio: float = 4.0            # negatives per positive
    ce_hard_frac: float = 0.7            # fraction of negatives drawn from the hardest (high name-cos)
    ce_max_pairs: int = 1_500_000        # cap on total training pairs
    ce_epochs: int = 2
    ce_lr: float = 2e-5
    ce_train_batch: int = 64
    ce_max_length: int = 128
    ce_val_pairs: int = 40_000           # held-out pairs to report train-time AUC/accuracy

    # ------------------------------------------------------------- validation
    val_fraction: float = 0.2
    beta: float = 0.5
