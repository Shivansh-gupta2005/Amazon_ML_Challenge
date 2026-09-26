#!/usr/bin/env python3
"""Command-line entry point for the Business Entity Resolution pipeline.

Examples
--------
# Train on dataset/train, report validation F0.5, then predict on dataset/test:
python src/run.py --data-dir dataset --output-dir output --model-dir artifacts

# Only train + validate (no test prediction):
python src/run.py --mode train

# Only predict using a previously saved model:
python src/run.py --mode predict

Run this from code/business_entity_resolution/ (so that `src/run.py` resolves).
"""
from __future__ import annotations

import argparse
import sys

# make `import ber` work when invoked as `python src/run.py` (sys.path[0] == src/)
from ber.config import Config
from ber import pipeline


def build_config(args) -> Config:
    cfg = Config()
    cfg.data_dir = args.data_dir
    cfg.output_dir = args.output_dir
    cfg.model_dir = args.model_dir
    if args.knn is not None:
        cfg.knn_name = args.knn
    if args.min_threshold is not None:
        cfg.min_threshold = args.min_threshold
    if args.val_fraction is not None:
        cfg.val_fraction = args.val_fraction
    if args.batch_size is not None:
        cfg.feature_batch_size = args.batch_size
    if args.max_train_entities is not None:
        cfg.max_train_entities = None if args.max_train_entities <= 0 else args.max_train_entities
    if args.max_features is not None:
        cfg.max_features_name = args.max_features
        cfg.max_features_addr = args.max_features
    if args.no_postal:
        cfg.use_postal_blocking = False
    if args.no_target_uniqueness:
        cfg.enforce_target_uniqueness = False
    if args.no_cold_country:
        cfg.use_cold_country_threshold = False
    if args.cold_threshold is not None:
        cfg.cold_threshold_override = args.cold_threshold
    if args.corroboration_delta is not None:
        cfg.corroboration_delta = args.corroboration_delta
    if args.use_embeddings:
        cfg.use_embeddings = True
    if args.embed_backend is not None:
        cfg.embed_backend = args.embed_backend
    if args.embed_model is not None:
        cfg.embed_model = args.embed_model
    if args.embed_device is not None:
        cfg.embed_device = args.embed_device
    if args.knn_embed is not None:
        cfg.knn_embed = args.knn_embed
    if args.embed_min_cosine is not None:
        cfg.embed_min_cosine = args.embed_min_cosine
    if args.use_cross_encoder:
        cfg.use_cross_encoder = True
    if args.cross_encoder_backend is not None:
        cfg.cross_encoder_backend = args.cross_encoder_backend
    return cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Business Entity Resolution pipeline")
    ap.add_argument("--mode", choices=["run", "train", "predict"], default="run",
                    help="run=train+validate+predict (default); train=train+validate; "
                         "predict=load saved model and predict on test")
    ap.add_argument("--data-dir", default="dataset")
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--model-dir", default="artifacts")
    ap.add_argument("--knn", type=int, default=None, help="nearest neighbours per source in blocking")
    ap.add_argument("--min-threshold", type=float, default=None, help="floor for decision threshold")
    ap.add_argument("--val-fraction", type=float, default=None, help="validation split fraction")
    ap.add_argument("--batch-size", type=int, default=None, help="candidate pairs featurised/scored per batch")
    ap.add_argument("--max-train-entities", type=int, default=None,
                    help="subsample this many Source-1 entities for training (0/negative = use all)")
    ap.add_argument("--max-features", type=int, default=None, help="TF-IDF vocab cap (name and address)")
    ap.add_argument("--no-postal", action="store_true", help="disable postal-code blocking")
    ap.add_argument("--no-target-uniqueness", action="store_true",
                    help="disable the one-target-per-Source1 constraint")
    ap.add_argument("--no-cold-country", action="store_true",
                    help="disable the leave-one-country-out threshold for unseen test countries (France)")
    ap.add_argument("--cold-threshold", type=float, default=None,
                    help="force the threshold used for unseen test countries (skips LOCO calibration)")
    ap.add_argument("--corroboration-delta", type=float, default=None,
                    help="accept borderline pairs within this margin below threshold if they share "
                         "a postal code or have high address cosine (0 = off, the default)")
    # v5 embedding stage
    ap.add_argument("--use-embeddings", action="store_true",
                    help="enable the local multilingual embedding blocker + feature (best on a GPU)")
    ap.add_argument("--embed-backend", choices=["sentence-transformers", "hashing"], default=None,
                    help="'sentence-transformers' (real, default) or 'hashing' (offline test stand-in)")
    ap.add_argument("--embed-model", default=None, help="sentence-transformers model id")
    ap.add_argument("--embed-device", default=None, help="cuda | cpu (default: auto-detect)")
    ap.add_argument("--knn-embed", type=int, default=None, help="embedding neighbours per source")
    ap.add_argument("--embed-min-cosine", type=float, default=None, help="embedding cosine floor for blocking")
    ap.add_argument("--use-cross-encoder", action="store_true",
                    help="rerank borderline pairs with a multilingual cross-encoder (experimental, GPU)")
    ap.add_argument("--cross-encoder-backend", choices=["sentence-transformers", "stub"], default=None,
                    help="'sentence-transformers' (real, default) or 'stub' (offline test stand-in)")
    args = ap.parse_args(argv)

    cfg = build_config(args)

    if args.mode == "predict":
        out = pipeline.predict_only(cfg)
        pipeline.log(f"done: {out}")
    elif args.mode == "train":
        import os
        from ber import model as model_mod, features as feat_mod
        m, t, metrics = pipeline.train_and_validate(cfg)
        os.makedirs(cfg.model_dir, exist_ok=True)
        model_mod.save_model(os.path.join(cfg.model_dir, "matcher.pkl"), m, t, feat_mod.FEATURE_NAMES,
                             extra={"cold_threshold": metrics.get("cold_threshold", t),
                                    "train_countries": metrics.get("train_countries", [])})
        pipeline.log(f"done: validation macro F0.5 = {metrics['val_macro_f_beta']:.4f}")
    else:
        metrics = pipeline.run_all(cfg)
        pipeline.log(f"done. validation macro F0.5 = {metrics['val_macro_f_beta']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
