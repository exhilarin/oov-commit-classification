"""Step 3: create the evaluation splits.

1. Random split (main benchmark): stratified 80/10/10 train/val/test, seed 42.
2. Held-out-repository split (generalisation test): every commit of the repositories
   listed in --heldout goes to `test_unseen_repos.csv`; models trained on the main
   train split never saw these projects, so this measures transfer to new codebases.
   Those repositories are removed from train/val/test of the random split.
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
HELDOUT = ["eslint/eslint", "tauri-apps/tauri", "typeorm/typeorm", "lerna/lerna", "webpack/webpack-cli"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=ROOT / "data/processed/commits_clean.csv", type=Path)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--heldout", nargs="*", default=HELDOUT)
    args = ap.parse_args()

    df = pd.read_csv(args.data, keep_default_na=False)
    out = args.data.parent
    unseen = df[df.repo.isin(args.heldout)]
    seen = df[~df.repo.isin(args.heldout)]

    train, rest = train_test_split(seen, test_size=0.2, stratify=seen.label, random_state=args.seed)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest.label, random_state=args.seed)
    for name, part in [("train", train), ("val", val), ("test", test), ("test_unseen_repos", unseen)]:
        part.to_csv(out / f"{name}.csv", index=False)
        print(f"{name:<18} {len(part):>6}  " + "  ".join(f"{k}={v}" for k, v in part.label.value_counts().items()))


if __name__ == "__main__":
    main()
