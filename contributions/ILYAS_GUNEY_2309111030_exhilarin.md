# İlyas Güney (2309111030, @exhilarin)

**Role:** implementation: data pipeline, models, benchmark and repository structure.

## What I did

### Data collection (`src/collect.py`, `configs/repos.yaml`)
* Selected 26 open-source repositories that follow Conventional Commits and checked that each one is
  MIT or Apache-2.0 licensed.
* Wrote the collector: commit-only bare clones (`git clone --bare --filter=tree:0`) so no source code is
  downloaded, `git log --no-merges` with custom field separators, bot detection from the author, and
  `sanitize_message`, which removes `Signed-off-by` / `Co-authored-by` trailers and masks e-mail addresses
  so no personal data is published. Result: 179,960 raw commits.

### Cleaning and labeling (`src/clean.py`)
* Parsing of the `type(scope)!: subject` header, alias mapping to 10 labels, and removal of the type and
  scope from the input text to prevent label leakage.
* Filters for bots, release/dependency scopes, URLs, issue references, hashes, non-English text,
  length, version-only subjects, ambiguous duplicates and exact duplicates.
* `cleaning_report.json`, which records how many rows every step removes (121,276 rows remain).

### Splits (`src/split.py`)
* Stratified 80/10/10 split and the held-out-repository test set (5 projects never seen in training).

### Models
* `train_classical.py`: Majority, Naive Bayes, Logistic Regression and Linear SVM on TF-IDF 1-2 grams,
  hyper-parameter chosen on validation macro-F1.
* `train_neural.py`: TextCNN (Kim 2014) and BiLSTM in PyTorch, trained from scratch, early stopping,
  gradient clipping, three seeds with mean/std.
* `train_transformer.py`: DistilBERT and ModernBERT fine-tuning with a plain PyTorch loop on CPU
  (AdamW, warm-up, per-batch padding trim, early stopping).
* `zero_shot_laya.py`: zero-shot evaluation of the Laya decision model (JEV alternative).

### Benchmark and reproducibility
* `common.py` (one evaluation function for every model), `benchmark.py` (results table, per-class F1,
  figures), `Makefile` (`make all` runs the full pipeline), pinned `requirements.txt`, fixed seeds.
* README, `docs/preprocessing.md` and `LICENSE`.

## My commits

All commits authored by `exhilarin` in this repository.
