# Data collection and preprocessing pipeline

The dataset is built by us from public Git history. No existing dataset is used.
Everything below is produced by three scripts and can be re-run with `make data splits`.

```
configs/repos.yaml ──► src/collect.py ──► data/raw/commits_raw.csv.gz
                                            │
                                            ▼
                        src/clean.py ──► data/processed/commits_clean.csv
                                         data/processed/cleaning_report.json
                                            │
                                            ▼
                        src/split.py ──► data/processed/{train,val,test,test_unseen_repos}.csv
```

## 1. Collection (`src/collect.py`)

* **Sources:** 26 open-source repositories that use the
  [Conventional Commits](https://www.conventionalcommits.org) convention, listed in
  `configs/repos.yaml`. Only MIT or Apache-2.0 licensed repositories were selected
  (license checked with the GitHub API on 2026-10-10).
* **How:** each repository is cloned as a *bare, commit-only* clone (`git clone --bare --filter=tree:0`),
  so no source code is downloaded. `git log --no-merges` reads up to 8,000 most recent commits of the
  default branch per repository.
* **Privacy:** author names and e-mail addresses are never written to disk. They are only used in memory
  to set the boolean `is_bot` column. Trailer lines that name people (`Signed-off-by:`, `Co-authored-by:`,
  `Reviewed-by:`, ...) are removed from every message and any remaining e-mail address is replaced by
  `<email>` (`sanitize_message` in `collect.py`).
* **Output:** `data/raw/commits_raw.csv.gz` with `repo, sha, date, is_bot, message`
  (179,960 commits). The `sha` makes every row traceable to its original commit.

## 2. Labeling and cleaning (`src/clean.py`)

A Conventional Commit header looks like `type(scope)!: subject`. The **type** becomes the label and
only the **subject** is kept as the input text:

```
header:      fix(router): handle errors from view transition readiness (#62535)
             (angular/angular, commit e9f2156)
label:       fix
removed:     the type "fix", the scope "(router)" and the PR reference "(#62535)"
input text:  "handle errors from view transition readiness"
```

Removing the type and the scope is essential. Otherwise the label would appear in the input
(label leakage) and the task would be trivial string matching.

| Step | Rule | Why | Rows after |
|---|---|---|---|
| 0 | raw commits | | 179,960 |
| 1 | drop commits whose author is a bot (dependabot, renovate, `[bot]`, ...) | templated, not human language | 158,161 |
| 2 | keep only headers matching `type(scope)!: subject` | no label otherwise | 144,526 |
| 3 | map type aliases to 10 labels (`feature→feat`, `bugfix→fix`, `doc→docs`, ...) and drop unknown types (`revert`, `release`, `wip`, ...) | one consistent label set | 138,580 |
| 4 | drop scopes `release`, `deps`, `deps-dev` | version bumps and dependency updates are near-identical templates | 135,676 |
| 5 | clean the subject: remove URLs, PR/issue references (`(#123)`, `closes #4`), commit hashes, trailing period, extra spaces; drop subjects with more than 10% non-ASCII characters | references carry no meaning about the change type; the task is English-only | 135,638 |
| 6 | keep subjects with 2–30 words, drop version-only subjects (`v1.2.3`) | too short to classify / not a description | 133,988 |
| 7 | drop texts that appear with **different** labels (case-insensitive) | genuinely ambiguous, would only add label noise | 129,002 |
| 8 | drop exact duplicate texts | avoids the same sentence appearing in both train and test | 121,276 |

Every number above is written to `data/processed/cleaning_report.json` by the script.

### Final label distribution (121,276 commits)

| label | meaning (Conventional Commits) | count | share |
|---|---|---|---|
| fix | a bug fix | 45,723 | 37.7% |
| docs | documentation only changes | 18,244 | 15.0% |
| feat | a new feature | 17,425 | 14.4% |
| chore | other maintenance changes | 15,316 | 12.6% |
| refactor | code change that neither fixes a bug nor adds a feature | 10,089 | 8.3% |
| test | adding or correcting tests | 5,157 | 4.3% |
| build | build system or external dependencies | 4,454 | 3.7% |
| ci | CI configuration and scripts | 2,321 | 1.9% |
| perf | performance improvement | 2,039 | 1.7% |
| style | formatting, no change in meaning | 508 | 0.4% |

The classes are heavily imbalanced (90:1 between `fix` and `style`), which is why the benchmark uses
**macro-F1** as its main metric.

## 3. Splits (`src/split.py`)

* **Random split (main benchmark):** stratified 80 / 10 / 10 → train 85,588, val 10,699, test 10,699 (seed 42).
  Validation is only used for model selection (hyper-parameters, early stopping).
* **Unseen-repositories split (generalisation):** all commits of `eslint/eslint`, `tauri-apps/tauri`,
  `typeorm/typeorm`, `lerna/lerna` and `webpack/webpack-cli` (14,290 commits) are removed from the random
  split and kept as `test_unseen_repos.csv`. It answers the question: *does a model trained on some projects
  work on a project it has never seen?* Each project has its own vocabulary and its own team habits for
  choosing a type.

## Known limitations

* Labels are chosen by the commit author, so they are noisy: the same change can honestly be called
  `fix`, `refactor` or `chore` by different people.
* Repositories differ in size, so some projects contribute more commits than others.
* Only the subject line is used; the commit body and the changed files are ignored on purpose to keep it a
  pure text-classification task.
