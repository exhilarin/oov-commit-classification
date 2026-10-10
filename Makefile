# Full pipeline:  make all      (data -> splits -> all models -> benchmark)
PY ?= python
SRC := cd src &&

.PHONY: all data splits classical neural transformers zeroshot benchmark clean

all: data splits classical neural transformers zeroshot benchmark

data:            ## 1. collect raw commits (needs git + internet) and clean them
	$(PY) src/collect.py
	$(PY) src/clean.py

splits:          ## 2. random 80/10/10 split + held-out-repository test split
	$(PY) src/split.py

classical:       ## 3a. Majority, Naive Bayes, Logistic Regression, Linear SVM
	$(SRC) $(PY) train_classical.py

neural:          ## 3b. TextCNN and BiLSTM from scratch, 3 seeds each
	$(SRC) $(PY) train_neural.py

transformers:    ## 3c. DistilBERT (baseline) and ModernBERT (new) fine-tuning
	$(SRC) $(PY) train_transformer.py --model distilbert --train-size 20000
	$(SRC) $(PY) train_transformer.py --model modernbert --train-size 20000

zeroshot:        ## 3d. Laya zero-shot decision model (JEV alternative)
	$(SRC) $(PY) zero_shot_laya.py

benchmark:       ## 4. results/benchmark.md + figures
	$(SRC) $(PY) benchmark.py

clean:
	rm -rf .cache models
