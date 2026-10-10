"""New method: zero-shot classification with Laya, an open-source (Apache-2.0) alternative to
JEV-style "System One" decision models. The model is NOT trained on our data: it only
receives the commit text and the 10 label names with their one-line definitions
(LABEL_DESCRIPTIONS in common.py, from the Conventional Commits guidelines) and returns
a typed choice with a probability per label.

Checkpoint: convaiinnovations/laya (ModernBERT-large encoder, 421M parameters).
Inference is slow on CPU, so by default we evaluate on a stratified random sample of each
test split (--n); the sample size is stored in the result file.
"""
import argparse

from sklearn.model_selection import train_test_split

from common import EVAL_SPLITS, LABEL_DESCRIPTIONS, Timer, evaluate, labels, load_split, save_result

QUESTION_ID = "commit_type"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=2000, help="stratified sample size per test split (0 = all)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from laya import Router  # imported here so the other scripts do not need laya installed

    label_list = labels()
    question = {QUESTION_ID: {
        "type": "choice",
        "instructions": "What kind of change does this git commit message describe?",
        "criteria": {l: LABEL_DESCRIPTIONS[l] for l in label_list},
    }}
    router = Router(device="cpu")
    scores, sizes = {}, {}
    with Timer() as t:
        for split in EVAL_SPLITS:
            df = load_split(split)
            if args.n and args.n < len(df):
                df, _ = train_test_split(df, train_size=args.n, stratify=df.label, random_state=args.seed)
            requests = [{"state": s, "questions": question, "model": "english"} for s in df.text]
            out = router.predict_batch(requests, batch_size=args.batch)
            pred = [r["answers"][QUESTION_ID]["choice"] for r in out]
            scores[split], sizes[split] = evaluate(df.label, pred, label_list), len(df)
    save_result("laya_zero_shot", "new method (zero-shot decision model, JEV alternative)",
                {"checkpoint": "convaiinnovations/laya", "training_examples": 0, "eval_sample_size": sizes,
                 "label_descriptions": question[QUESTION_ID]["criteria"]}, scores, 0.0,
                {"inference_seconds": round(t.seconds, 1)})


if __name__ == "__main__":
    main()
