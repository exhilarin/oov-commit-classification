"""Step 2: turn raw commit messages into a labeled, cleaned dataset.

Input : data/raw/commits_raw.csv.gz  (from src/collect.py)
Output: data/processed/commits_clean.csv   all cleaned, labeled rows
        data/processed/cleaning_report.json how many rows every step removed

The label is the Conventional Commits type in the header, e.g.
    "fix(router): handle trailing slash (#1234)"  ->  label=fix, text="handle trailing slash"
The type and the scope are removed from the text, so a model must infer the
change type from the description alone. See docs/preprocessing.md for the reasoning.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

HEADER_RE = re.compile(r"^(?P<type>[A-Za-z]+)(?:\((?P<scope>[^)]*)\))?(?P<bang>!)?\s*:\s*(?P<subject>.+)$")
ALIASES = {
    "feat": "feat", "feature": "feat", "features": "feat",
    "fix": "fix", "fixes": "fix", "bugfix": "fix", "bug": "fix", "hotfix": "fix",
    "docs": "docs", "doc": "docs", "documentation": "docs",
    "refactor": "refactor", "refactoring": "refactor",
    "test": "test", "tests": "test", "testing": "test",
    "perf": "perf", "performance": "perf",
    "build": "build", "ci": "ci",
    "chore": "chore", "chores": "chore",
    "style": "style",
}
DROP_SCOPES = {"release", "deps", "deps-dev", "dev-deps"}   # templated, mostly automated commits
PR_REF_RE = re.compile(r"\s*\((?:#|GH-|gh-)\d+\)|\s*(?:closes|fixes|resolves|refs?)\s+#\d+|(?<!\w)#\d+", re.I)
URL_RE = re.compile(r"https?://\S+")
SHA_RE = re.compile(r"\b[0-9a-f]{7,40}\b")
VERSION_ONLY_RE = re.compile(r"^(?:release|bump|version|publish)?\s*v?\d+\.\d+(?:\.\d+)?(?:[-+.\w]*)?$", re.I)


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def clean_subject(s: str) -> str:
    s = s.split("\\n")[0]            # a few subjects contain a literal "\n" followed by trailer text
    s = EMAIL_RE.sub(" ", s).replace("<email>", " ")
    s = URL_RE.sub(" ", s)
    s = PR_REF_RE.sub(" ", s)
    s = SHA_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip().rstrip(".").strip()
    return s


def non_ascii_ratio(s: str) -> float:
    return sum(ord(c) > 127 for c in s) / max(len(s), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=ROOT / "data/raw/commits_raw.csv.gz", type=Path)
    ap.add_argument("--out", default=ROOT / "data/processed/commits_clean.csv", type=Path)
    ap.add_argument("--min-class", default=500, type=int, help="drop labels with fewer examples")
    ap.add_argument("--min-words", default=2, type=int)
    ap.add_argument("--max-words", default=30, type=int)
    args = ap.parse_args()

    report = {}
    df = pd.read_csv(args.raw, dtype={"message": str}, keep_default_na=False)
    report["0_raw"] = len(df)

    df = df[df.is_bot == 0]
    report["1_after_removing_bot_authors"] = len(df)

    df = df.assign(header=df.message.str.split("\n").str[0].str.strip())
    m = df.header.str.extract(HEADER_RE)
    df = pd.concat([df, m], axis=1).dropna(subset=["type", "subject"])
    report["2_after_keeping_conventional_headers"] = len(df)

    df["label"] = df["type"].str.lower().map(ALIASES)
    df = df.dropna(subset=["label"])
    report["3_after_mapping_known_types"] = len(df)

    df["scope"] = df["scope"].fillna("").str.strip().str.lower()
    df = df[~df.scope.isin(DROP_SCOPES)]
    report["4_after_dropping_release_and_dependency_scopes"] = len(df)

    df["text"] = df.subject.map(clean_subject)
    df = df[df.text.map(non_ascii_ratio) <= 0.1]
    report["5_after_dropping_non_english"] = len(df)

    n_words = df.text.str.split().str.len()
    df = df[(n_words >= args.min_words) & (n_words <= args.max_words) & ~df.text.str.match(VERSION_ONLY_RE)]
    report["6_after_length_and_version_filters"] = len(df)

    # duplicates: same text with one label -> keep one; same text with several labels -> ambiguous, drop all
    key = df.text.str.lower()
    labels_per_text = df.groupby(key).label.nunique()
    ambiguous = set(labels_per_text[labels_per_text > 1].index)
    df = df[~key.isin(ambiguous)]
    report["7_after_dropping_ambiguous_duplicates"] = len(df)
    df = df.loc[~df.text.str.lower().duplicated()]
    report["8_after_dropping_exact_duplicates"] = len(df)

    counts = df.label.value_counts()
    keep = counts[counts >= args.min_class].index
    report["dropped_rare_labels"] = {k: int(v) for k, v in counts.items() if k not in keep}
    df = df[df.label.isin(keep)]
    report["9_final"] = len(df)
    report["label_counts"] = {k: int(v) for k, v in df.label.value_counts().items()}
    report["repo_counts"] = {k: int(v) for k, v in df.repo.value_counts().items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df["breaking"] = df.bang.fillna("").eq("!").astype(int)
    df[["repo", "sha", "label", "scope", "breaking", "text"]].to_csv(args.out, index=False)
    (args.out.parent / "cleaning_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
