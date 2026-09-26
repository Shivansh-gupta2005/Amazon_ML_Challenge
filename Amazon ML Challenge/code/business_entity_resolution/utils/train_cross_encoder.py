#!/usr/bin/env python3
"""v6 -- fine-tune a cross-encoder on the challenge's OWN labels.

WHY (from the data diagnostic): matching on exact name has only ~0.41 precision --
lots of DIFFERENT businesses share a name (branches/franchises). The address is what
decides "same or not". An off-the-shelf model doesn't know this task; a cross-encoder
FINE-TUNED on the ground-truth pairs learns exactly the same-name/different-address
disambiguation. Hard negatives -- high name-similarity NON-matches -- are the core
teaching signal, so the model must look past the name.

COMPLIANCE: trains only on the PROVIDED train labels; base model is Apache-2.0 and
<8B params; runs locally. No external data. (Needs a GPU to train in reasonable time.)

Pipeline reuse: the same normalisation, TF-IDF index and blocking as the main pipeline
build the candidate pairs, so training text matches inference text exactly (name | addr
| country, via rerank.compose_texts).

Usage (on the GPU box, after `pip install -r requirements-embeddings.txt`):
    python3 utils/train_cross_encoder.py \
        --data-dir "<dataset>" --out-dir "$HOME/ce_model"
Then run the pipeline with:
    python3 src/run.py ... --use-cross-encoder --cross-encoder-model "$HOME/ce_model" \
        --cross-encoder-band 0.30

Test the pair construction offline (no GPU / no model):
    python3 utils/train_cross_encoder.py --data-dir dataset --out-dir /tmp/ce --dry-run
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from ber import blocking, io_utils, pipeline, rerank            # noqa: E402
from ber.config import Config, RANDOM_SEED                      # noqa: E402

_t0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time()-_t0:6.0f}s] {msg}", flush=True)


def build_training_pairs(cfg: Config, ctx, gt):
    """Return (texts_a, texts_b, labels) -- positives (GT + blocking-missed) and a mix
    of HARD (high name-cosine) and random negatives, as name|addr|country text pairs."""
    rng = np.random.default_rng(RANDOM_SEED)
    k = min(cfg.ce_train_entities, ctx.n1)
    all_rows = np.arange(ctx.n1)
    sample_rows = np.sort(rng.choice(all_rows, k, replace=False)) if ctx.n1 > k else all_rows
    s1n_s = ctx.s1n.loc[ctx.s1n["row"].isin(set(sample_rows.tolist()))]

    log(f"blocking {len(s1n_s)} sampled Source-1 entities to mine pairs ...")
    cand = blocking.generate_candidates(cfg, ctx.index, s1n_s, ctx.s2n, ctx.s3n,
                                        ctx.name_key_by_row, embed_mat=ctx.embed_mat)
    y = pipeline.label_pairs(ctx, cand, gt)

    # positives: blocked true pairs + the true pairs blocking missed (harder positives)
    inj = pipeline.injected_only(ctx, cand, gt, sample_rows)
    pos_s1 = np.concatenate([cand["s1_row"][y == 1], inj["s1_row"]]).astype(np.int64)
    pos_cn = np.concatenate([cand["cand_row"][y == 1], inj["cand_row"]]).astype(np.int64)
    n_pos = len(pos_s1)

    # negatives: blocked non-matches; take the hardest (highest name cosine) + some random
    nmask = y == 0
    neg_s1 = cand["s1_row"][nmask].astype(np.int64)
    neg_cn = cand["cand_row"][nmask].astype(np.int64)
    neg_cos = cand["name_cos"][nmask]
    n_neg = int(min(len(neg_s1), round(cfg.ce_neg_ratio * n_pos)))
    n_hard = int(round(cfg.ce_hard_frac * n_neg))
    order = np.argsort(-neg_cos)                       # hardest first
    hard_idx = order[:n_hard]
    rest = order[n_hard:]
    n_rand = min(n_neg - n_hard, len(rest))
    rand_idx = rng.choice(rest, size=n_rand, replace=False) if n_rand > 0 else np.zeros(0, int)
    neg_idx = np.concatenate([hard_idx, rand_idx])

    sel_s1 = np.concatenate([pos_s1, neg_s1[neg_idx]])
    sel_cn = np.concatenate([pos_cn, neg_cn[neg_idx]])
    labels = np.concatenate([np.ones(n_pos, np.float32), np.zeros(len(neg_idx), np.float32)])

    if len(labels) > cfg.ce_max_pairs:                # cap + shuffle
        keep = rng.choice(len(labels), cfg.ce_max_pairs, replace=False)
        sel_s1, sel_cn, labels = sel_s1[keep], sel_cn[keep], labels[keep]

    cn = ctx.arr["core_name"]; ad = ctx.arr["norm_addr"]; co = ctx.arr["country_key"]
    ta = rerank.compose_texts(cn[sel_s1], ad[sel_s1], co[sel_s1])
    tb = rerank.compose_texts(cn[sel_cn], ad[sel_cn], co[sel_cn])
    log(f"pairs built: total={len(labels)}  pos={int(labels.sum())}  "
        f"neg={int((labels==0).sum())}  (hard≈{n_hard}, random≈{n_rand})")
    return ta, tb, labels


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True, help="where to save the fine-tuned model")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--train-entities", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="build + report pairs only; no training")
    args = ap.parse_args(argv)

    cfg = Config(); cfg.data_dir = args.data_dir
    if args.base_model: cfg.ce_base_model = args.base_model
    if args.train_entities: cfg.ce_train_entities = args.train_entities
    if args.epochs: cfg.ce_epochs = args.epochs

    paths = io_utils.resolve_source_paths(cfg, "train")
    log(f"loading train from {os.path.dirname(paths['source1'])}")
    s1 = io_utils.read_source(paths["source1"])
    s2 = io_utils.read_source(paths["source2"])
    s3 = io_utils.read_source(paths["source3"])
    gt = io_utils.read_ground_truth(paths["ground_truth"])
    ctx = pipeline.Context(cfg, s1, s2, s3)            # embeddings off -> CPU-only pair mining
    del s1, s2, s3

    ta, tb, labels = build_training_pairs(cfg, ctx, gt)

    # deterministic train/val split
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(labels))
    n_val = min(cfg.ce_val_pairs, len(labels) // 5)
    va, tr = idx[:n_val], idx[n_val:]

    if args.dry_run:
        log("DRY RUN -- sample composed pairs:")
        for j in list(tr[:3]) + list(np.where(labels == 0)[0][:2]):
            print(f"  y={labels[j]:.0f}  A={ta[j][:70]!r}  B={tb[j][:70]!r}")
        log(f"train={len(tr)}  val={len(va)}  pos_rate={labels.mean():.3f}")
        log("dry run OK (skipped model download + training).")
        return 0

    # ---- fine-tune (needs sentence-transformers + torch + a GPU) ----
    from sentence_transformers import CrossEncoder, InputExample
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
    from torch.utils.data import DataLoader

    log(f"loading base cross-encoder: {cfg.ce_base_model}")
    model = CrossEncoder(cfg.ce_base_model, num_labels=1, max_length=cfg.ce_max_length)
    train_ex = [InputExample(texts=[ta[i], tb[i]], label=float(labels[i])) for i in tr]
    loader = DataLoader(train_ex, shuffle=True, batch_size=cfg.ce_train_batch)
    evaluator = CEBinaryClassificationEvaluator(
        [[ta[i], tb[i]] for i in va], [float(labels[i]) for i in va], name="val")
    warmup = int(0.1 * len(loader) * cfg.ce_epochs)
    os.makedirs(args.out_dir, exist_ok=True)
    log(f"fine-tuning: {len(train_ex)} pairs, {cfg.ce_epochs} epochs, lr={cfg.ce_lr}")
    model.fit(train_dataloader=loader, evaluator=evaluator, epochs=cfg.ce_epochs,
              warmup_steps=warmup, optimizer_params={"lr": cfg.ce_lr},
              output_path=args.out_dir, use_amp=True)
    model.save(args.out_dir)
    log(f"saved fine-tuned cross-encoder -> {args.out_dir}")
    log("now run: src/run.py ... --use-cross-encoder "
        f"--cross-encoder-model {args.out_dir} --cross-encoder-band 0.30")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
