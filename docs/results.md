# Results and error analysis

All numbers come from `results/*.json` and `results/benchmark.md`. Primary metric: macro-F1 over 10 classes.

## 1. Overview

| Model | Group | Test macro-F1 | Unseen-repos macro-F1 | Drop |
|---|---|---|---|---|
| Majority class | baseline | 0.055 | 0.051 | |
| Naive Bayes | baseline | 0.478 | 0.301 | −17.7 |
| Logistic Regression | baseline | **0.508** | 0.351 | −15.7 |
| Linear SVM | baseline | 0.506 | 0.336 | −17.0 |
| TextCNN | baseline | 0.485 | 0.345 | −14.0 |
| BiLSTM | baseline | 0.491 | 0.337 | −15.4 |
| DistilBERT | baseline | 0.435 | 0.347 | −8.8 |
| ModernBERT-base | new | 0.486 | **0.382** | −10.4 |
| Laya (zero-shot) | new | 0.188 | 0.158 | −3.0 |

### What the numbers say

**1. Every trained model beats the majority baseline by a factor of 8 to 9**, so the commit subject alone carries
real information about the change type.

**2. On the random split the simple linear models win.** Logistic Regression (0.508) and Linear SVM (0.506) are
ahead of TextCNN (0.485), BiLSTM (0.491) and the transformers. The reasons:
* Commit subjects are very short (6 words on average) and their type is often signalled by a few keywords. The words
  the Logistic Regression model weights most per class are:
  * `perf`: optimize, cache, avoid, faster, lazy
  * `docs`: clarify, guide, example, document
  * `ci`: job, workflow, workflows, step, actions
  * `test`: test, tests, cover, assert, fixture
  * `refactor`: extract, simplify, move, refactor
  * `fix`: should, when, correctly, handle, preserve, properly

  TF-IDF n-grams capture exactly these signals.
* The classical models used all 85,588 training commits; the transformers only 20,000 (CPU budget) and their
  validation score was still rising when training stopped.

**3. On unseen repositories the ranking changes: ModernBERT is the best (0.382).** Every model loses accuracy on
projects it never saw, but by very different amounts:
* the TF-IDF and from-scratch models lose 14–18 points: they partly memorise project-specific words
  (package names, internal tool names);
* the pretrained transformers lose only 9–10 points: their pre-trained knowledge of English and code transfers to
  new projects. ModernBERT was pre-trained on code as well as text, and it generalises best, with less than a
  quarter of the training data.

For a real tool, used on a repository it was not trained on, the unseen-repository score is the more honest number.

**4. Zero-shot is far behind.** Laya reaches 0.188 with no training data (3.4× the majority baseline), but it
predicts `refactor` for 836 of 2,000 test commits and `fix` only 64 times, although `fix` is 38% of the data. The
definition of `refactor` ("a code change that neither fixes a bug nor adds a feature") works as a catch-all. The
way developers actually use the ten types has to be learned from labeled examples.

## 2. Which classes are hard (Logistic Regression, random test split)

| class | F1 | recall | most often confused with |
|---|---|---|---|
| fix | 0.75 | 0.83 | feat 6% |
| docs | 0.71 | 0.71 | fix 14% |
| build | 0.65 | 0.56 | fix 17%, chore 12% |
| test | 0.60 | 0.56 | fix 26% |
| feat | 0.57 | 0.54 | **fix 31%** |
| chore | 0.52 | 0.48 | fix 22%, docs 11% |
| refactor | 0.50 | 0.46 | **fix 32%** |
| ci | 0.40 | 0.32 | chore 25%, fix 21% |
| perf | 0.30 | 0.21 | **fix 55%** |
| style | 0.07 | 0.04 | **fix 58%** |

![confusion matrix](../results/figures/confusion_logistic_regression.png)

**`fix` is the attractor.** It is the largest class (38%), and almost every class leaks into it. This is the class
imbalance working against the rare classes, and the reason why accuracy would hide the problem.

## 3. Error examples (true label → predicted label)

**perf → fix:** the change improves speed, but the wording sounds like a correction.
* "reduce url to/from path conversions"
* "skip expanding exclude globs"
* "prewarm the browser while the Vite server starts"

**style → fix:** formatting changes described like bug fixes. One message even starts with the word "fix".
* "fix input addon misplace when zoom up"
* "use flex layout"
* "remove console log, update error message"

**ci → chore:** CI maintenance looks like general maintenance.
* "increase node compat test timeout on Mac Intel"
* "fix nightly publish"
* "rm backup action version comment"

**refactor → fix / feat → fix:**
* "transformHArgs -> transformVNodeArgs" (a rename, no verb at all)
* "ensure mock file system can be loaded in the browser" (labeled refactor by its author)
* "allow for preview container to be customized" (labeled feat)

Many of these errors are not really model mistakes: a human would also hesitate, and some labels are arguably
wrong. For example, "fix input addon misplace" was labeled `style`, and "remove console log, update error
message" could honestly be `chore`, `style` or `fix`. **Label noise sets a ceiling on any model trained on these
labels.**

## 4. Limitations and what we would do next

* **More input than the subject:** the list of changed files would resolve many errors (a change touching only
  `.md` files is `docs`, only `.github/workflows` is `ci`, only `*.test.*` is `test`).
* **GPU training on the full data:** the transformers saw less than a quarter of the training set; with all of it
  they would very likely move above the classical models on the random split as well.
* **Label noise:** use agreement between several models to find and relabel inconsistent examples, or merge
  classes that are inherently ambiguous (`style`/`chore`).
* **Laya:** better label descriptions could help, but tuning them while looking at the test set would be cheating;
  it would need its own validation set.
