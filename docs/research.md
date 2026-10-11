# Research notes: problem definition and design decisions

## 1. The problem

**Task:** given the description of a code change (a git commit subject line), predict the *type* of the
change: is it a new feature, a bug fix, documentation, a refactoring, a test, a build/CI change, a performance
improvement, a style change or general maintenance?

```
"handle platform destruction during rendering"   -> fix       (angular/angular-cli)
"Add remote image support"                       -> feat      (gatsbyjs/gatsby)
"remove dead code and redundant logic"           -> refactor  (ant-design/ant-design)
"reduce redundant array iterations"              -> perf      (ant-design/ant-design)
```

It is a **10-class, single-label text classification** problem on short, technical English text.

### Why it is useful

* **Changelog and release automation:** tools such as semantic-release decide the next version number
  (major / minor / patch) from commit types. A classifier can suggest the type when the developer did not
  write one, or warn when the written type looks wrong.
* **Repository analytics:** "what share of our work is bug fixing?" can be answered for repositories that do
  not follow any convention.
* **Code review triage:** a `fix` and a `refactor` need different kinds of review.

### Why it is a new problem for this course

The course presentations covered sentiment analysis, spam detection, topic classification, intent
classification, content moderation, authorship attribution, medical, legal and emotion classification.
Classifying **software change intent** from commit messages is none of these. It is closest to intent
classification, but the "intent" is an engineering activity and the language is a technical jargon in which
the same word changes meaning (`"update docs"` → docs, `"update parser to handle null"` → fix).

## 2. Why this dataset

* We **built it ourselves** from the public Git history of 26 MIT / Apache-2.0 licensed projects, so no other
  group can have the same data and the whole collection and cleaning is a script in this repository
  (see `docs/preprocessing.md`).
* Labels come for free and at scale: projects that follow the
  [Conventional Commits](https://www.conventionalcommits.org) convention write the type at the start of every
  commit header (`fix(router): ...`). We strip the type and the scope from the text, so the label is no longer
  visible to the model.
* The result is large (121,276 commits) and realistic: written by thousands of different developers.

## 3. Label set

The 10 types of the Conventional Commits / Angular guidelines:

| label | definition |
|---|---|
| feat | a new feature |
| fix | a bug fix |
| docs | documentation only changes |
| style | changes that do not affect the meaning of the code (formatting, white-space) |
| refactor | a code change that neither fixes a bug nor adds a feature |
| perf | a code change that improves performance |
| test | adding missing tests or correcting existing tests |
| build | changes that affect the build system or external dependencies |
| ci | changes to CI configuration files and scripts |
| chore | other maintenance changes that do not modify source or test files |

**Expected difficulty.** Some pairs are hard even for humans: `refactor` vs `chore` vs `style`, `build` vs `ci`
vs `chore`, and `fix` vs `perf`. Labels are written by the commit authors, so there is natural label noise.

## 4. Choice of evaluation metric

The class distribution is very skewed: `fix` is 37.7% of the data and `style` only 0.4% (90:1).

* **Accuracy is misleading here.** A model that always answers `fix` already gets 38% accuracy while being
  useless for the 9 other classes.
* **Macro-F1** (the unweighted mean of the per-class F1 scores) gives every class the same weight, so a model
  only scores well if it also recognises rare classes such as `perf`, `ci` or `style`. That matches the use
  case: a changelog tool that never detects `perf` or `docs` is broken.
* We therefore rank models by **macro-F1** and also report accuracy, weighted-F1, per-class F1 and the
  confusion matrix. AUC is not used as the main metric because it does not measure the actual decision of a
  multi-class model.

## 5. Choice of methods

Following the instructions, the vanilla methods from the lectures are **baselines** and anything beyond them
is a **new method**.

| Group | Method | Architecture vs. concrete model |
|---|---|---|
| baseline | Majority class | lower bound |
| baseline | Multinomial Naive Bayes | TF-IDF word 1-2 grams |
| baseline | Logistic Regression | TF-IDF word 1-2 grams |
| baseline | Linear SVM | TF-IDF word 1-2 grams |
| baseline | TextCNN (Kim, 2014) | CNN architecture, our own model trained from scratch |
| baseline | BiLSTM | LSTM architecture, our own model trained from scratch |
| baseline | DistilBERT | architecture: BERT (Transformer encoder); model: `distilbert-base-uncased` (66M) |
| **new** | ModernBERT | architecture: Transformer encoder (2024, RoPE, local/global attention, trained on code + text); model: `answerdotai/ModernBERT-base` (149M) |
| **new** | Laya (zero-shot) | open-source alternative to JEV "System One" decision models; model: `convaiinnovations/laya` (ModernBERT-large, 421M); receives only the label definitions above, no training data |

**Why ModernBERT:** commit messages are full of code tokens (`useEffect`, `tsconfig`, `--watch`). ModernBERT was
pre-trained on a large amount of code in addition to text, so it should understand this vocabulary better
than BERT models trained on books and Wikipedia.

**Why Laya:** it shows what is possible with zero labeled data, which is the realistic situation for a
repository that does not follow Conventional Commits.

## 6. Extra experiment: unseen repositories

Every project has its own vocabulary and its own habits for choosing a type. To measure whether models learn
the *task* and not the *projects*, five repositories are completely held out and used only as a second test
set (see `docs/preprocessing.md`).

## References

* Conventional Commits 1.0.0 specification, https://www.conventionalcommits.org
* Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP.
* Sanh, V. et al. (2019). DistilBERT, a distilled version of BERT. arXiv:1910.01108.
* Warner, B. et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder (ModernBERT). arXiv:2412.13663.
* Laya, https://github.com/NandhaKishorM/laya (Apache-2.0)
