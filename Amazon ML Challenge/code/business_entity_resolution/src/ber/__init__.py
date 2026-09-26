"""Business Entity Resolution pipeline.

A modular, precision-oriented entity-resolution pipeline that links records from
three independent business data sources. Source 1 is the deduplicated reference
source; for every Source 1 record we find its matching records in Source 2 and
Source 3.

Modules
-------
config     : hyper-parameters and paths
io_utils   : robust TSV reading / writing and ground-truth parsing
normalize  : name / address normalisation (abbreviations, transliteration, etc.)
blocking   : candidate generation (TF-IDF char n-gram KNN + postal blocking)
features   : pairwise feature engineering
model      : LightGBM pairwise matcher (train / predict) + threshold tuning
resolve    : precision-oriented assignment (target-uniqueness constraint)
evaluate   : F0.5 metric (macro over Source 1 entities) and validation split
pipeline   : end-to-end orchestration (train + predict)
"""

__version__ = "1.0.0"
