"""Shared helpers: loading the splits, the label set and one evaluation function used by every model."""
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
RESULTS = ROOT / "results"
EVAL_SPLITS = ["test", "test_unseen_repos"]

# Label definitions, taken from the Conventional Commits / Angular commit guidelines.
# Used for documentation and as the label descriptions for the zero-shot decision model.
LABEL_DESCRIPTIONS = {
    "feat": "a new feature",
    "fix": "a bug fix",
    "docs": "documentation only changes",
    "style": "changes that do not affect the meaning of the code (formatting, white-space)",
    "refactor": "a code change that neither fixes a bug nor adds a feature",
    "perf": "a code change that improves performance",
    "test": "adding missing tests or correcting existing tests",
    "build": "changes that affect the build system or external dependencies",
    "ci": "changes to continuous integration configuration files and scripts",
    "chore": "other maintenance changes that do not modify source or test files",
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_split(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv", keep_default_na=False)


def labels() -> list[str]:
    return sorted(load_split("train").label.unique())


def evaluate(y_true, y_pred, label_list) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=label_list, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=label_list, average="weighted", zero_division=0),
        "per_class": classification_report(y_true, y_pred, labels=label_list, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=label_list).tolist(),
    }


def save_result(model_name: str, family: str, params: dict, scores: dict, train_seconds: float, extra=None):
    """Write results/<model_name>.json; benchmark.py aggregates these files."""
    RESULTS.mkdir(exist_ok=True)
    out = {"model": model_name, "family": family, "params": params, "train_seconds": round(train_seconds, 1),
           "labels": labels(), "scores": scores, **(extra or {})}
    (RESULTS / f"{model_name}.json").write_text(json.dumps(out, indent=2))
    msg = "  ".join(f"{s}: macroF1={v['macro_f1']:.4f} acc={v['accuracy']:.4f}" for s, v in scores.items())
    print(f"[{model_name}] {msg}")


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.seconds = time.time() - self.t
