# Nuri Tayfun Akbeniz (210911043, @akbenizt)

**Role:** research: problem definition, label set, evaluation metric and method choices.

## What I did

* **Problem selection.** Compared candidate text-classification problems against the grading criteria
  (problem novelty, data novelty) and proposed classifying the type of a code change from its commit
  message, a problem that is not among the tasks covered in the course presentations.
* **Data source research.** Proposed using repositories that follow the Conventional Commits convention,
  because the type at the start of every header gives a label for free and at scale, and identified the
  label leakage risk (the type must be removed from the text).
* **Label set.** Defined the 10 classes and their meanings from the Conventional Commits / Angular
  guidelines; these definitions are also the label descriptions given to the zero-shot Laya model.
* **Metric choice.** Analysed the class imbalance (fix 37.7% vs style 0.4%) and argued for macro-F1 as the
  main metric instead of accuracy; documented why accuracy and AUC are not suitable as the main ranking metric.
* **Method selection.** Separated the baselines (vanilla methods from the lectures) from the new methods,
  and distinguished architectures from concrete models (BERT → distilbert-base-uncased; Transformer
  encoder → ModernBERT-base). Researched JEV and found Laya as an open-source alternative.
* **Generalisation experiment.** Proposed testing on held-out repositories to check whether models learn
  the task or the individual projects.

All of this is written up in [`docs/research.md`](../docs/research.md).
