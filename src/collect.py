"""Step 1: collect raw commit messages from the repositories in configs/repos.yaml.

Each repository is cloned as a bare, commit-only clone (`--filter=tree:0`), so no
source files are downloaded. We then read every non-merge commit on the default
branch with `git log`.

Output: data/raw/commits_raw.csv.gz with columns
    repo, sha, date, is_bot, message
Author names and e-mails are NOT stored; we only keep a boolean `is_bot` flag. Trailer lines
that name people (Signed-off-by, Co-authored-by, Reviewed-by, ...) are removed from the
message and any remaining e-mail address is replaced by <email>.
"""
import argparse
import csv
import gzip
import re
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BOT_RE = re.compile(r"\[bot\]|dependabot|renovate|github-actions|greenkeeper|semantic-release-bot|-bot\b", re.I)
SEP_REC, SEP_FIELD = "\x1e", "\x1f"
TRAILER_RE = re.compile(r"^\s*[A-Za-z][A-Za-z-]*-by:.*$", re.I | re.M)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def sanitize_message(msg: str) -> str:
    """Remove person-identifying trailers and mask e-mail addresses."""
    msg = TRAILER_RE.sub("", msg)
    return EMAIL_RE.sub("<email>", msg).strip()


def git(*args, cwd=None):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def clone_or_update(name: str, cache: Path) -> Path:
    target = cache / (name.replace("/", "__") + ".git")
    if target.exists():
        git("fetch", "--quiet", "origin", cwd=target)
    else:
        git("clone", "--quiet", "--bare", "--filter=tree:0", f"https://github.com/{name}.git", str(target))
    return target


def read_commits(repo_dir: Path, limit: int):
    fmt = SEP_FIELD.join(["%H", "%cI", "%an", "%ae", "%B"]) + SEP_REC
    out = git("log", "--no-merges", f"--max-count={limit}", f"--format={fmt}", "HEAD", cwd=repo_dir)
    for rec in out.split(SEP_REC):
        rec = rec.strip("\n")
        if not rec:
            continue
        sha, date, author, email, body = rec.split(SEP_FIELD, 4)
        yield sha, date, bool(BOT_RE.search(author) or BOT_RE.search(email)), sanitize_message(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=ROOT / "configs/repos.yaml", type=Path)
    ap.add_argument("--cache", default=ROOT / ".cache/repos", type=Path, help="where bare clones are kept (git-ignored)")
    ap.add_argument("--limit", default=8000, type=int, help="max commits per repository")
    ap.add_argument("--out", default=ROOT / "data/raw/commits_raw.csv.gz", type=Path)
    args = ap.parse_args()

    repos = yaml.safe_load(args.config.read_text())["repos"]
    args.cache.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with gzip.open(args.out, "wt", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["repo", "sha", "date", "is_bot", "message"])
        for r in repos:
            path = clone_or_update(r["name"], args.cache)
            n = 0
            for sha, date, is_bot, msg in read_commits(path, args.limit):
                w.writerow([r["name"], sha, date, int(is_bot), msg])
                n += 1
            total += n
            print(f"{r['name']:<40} {n:>6} commits")
    print(f"total: {total} commits -> {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
