"""Classical baselines from the course (Week 2): Majority class, Multinomial Naive Bayes,
Logistic Regression and Linear SVM, all on TF-IDF word uni+bigram features.

The one hyper-parameter of each model (alpha for NB, C for LR/SVM) is selected on the
validation split by macro-F1; the selected model is then evaluated once on the test splits.
"""
import argparse

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from common import EVAL_SPLITS, Timer, evaluate, labels, load_split, save_result, seed_everything

MODELS = {
    "majority_class": (lambda _: DummyClassifier(strategy="most_frequent"), [None]),
    "naive_bayes": (lambda a: MultinomialNB(alpha=a), [0.01, 0.03, 0.1, 0.3, 1.0]),
    "logistic_regression": (lambda c: LogisticRegression(C=c, max_iter=2000), [0.3, 1, 3, 10, 30]),
    "linear_svm": (lambda c: LinearSVC(C=c), [0.03, 0.1, 0.3, 1, 3]),
}


def features():
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)

    train, val = load_split("train"), load_split("val")
    label_list = labels()
    for name in args.models:
        build, grid = MODELS[name]
        best = None
        with Timer() as t:
            for hp in grid:
                pipe = make_pipeline(features(), build(hp)).fit(train.text, train.label)
                f1 = f1_score(val.label, pipe.predict(val.text), average="macro")
                if best is None or f1 > best[0]:
                    best = (f1, hp, pipe)
        val_f1, hp, pipe = best
        scores = {s: evaluate(load_split(s).label, pipe.predict(load_split(s).text), label_list) for s in EVAL_SPLITS}
        save_result(name, "baseline (classical)", {"features": "tfidf word 1-2gram", "hyperparameter": hp,
                                                   "val_macro_f1": round(val_f1, 4)}, scores, t.seconds)


if __name__ == "__main__":
    main()
