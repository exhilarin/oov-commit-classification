"""Fine-tune a pretrained Transformer encoder for commit-type classification.

* distilbert-base-uncased   -> baseline (BERT architecture, distilled checkpoint covered in Week 3)
* answerdotai/ModernBERT-base -> new method (2024 encoder: RoPE positions, alternating local/global
                                 attention, GeGLU, trained on code + text)

Plain PyTorch training loop on CPU: AdamW, linear warm-up/decay, early stopping on
validation macro-F1. `--train-size` lets you subsample the training split (stratified)
when compute is limited, and `--val-size` the validation sample used for early stopping;
both values are recorded in the result file.
"""
import argparse
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common import EVAL_SPLITS, ROOT, Timer, evaluate, labels, load_split, save_result, seed_everything

PRESETS = {
    "distilbert": ("distilbert-base-uncased", "baseline (pretrained transformer)"),
    "modernbert": ("answerdotai/ModernBERT-base", "new method (pretrained transformer)"),
}


def batches(enc, y, bs, shuffle):
    idx = np.random.permutation(len(y)) if shuffle else np.arange(len(y))
    for i in range(0, len(y), bs):
        j = idx[i:i + bs]
        b = {k: v[j] for k, v in enc.items()}
        width = int(b["attention_mask"].sum(1).max())   # trim padding to the longest message in this batch
        yield {k: v[:, :width] for k, v in b.items()}, y[j]


def predict(model, enc, bs=128):
    model.eval()
    out = []
    with torch.no_grad():
        for b, _ in batches(enc, torch.zeros(len(enc["input_ids"]), dtype=torch.long), bs, False):
            out.append(model(**b).logits.argmax(-1))
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(PRESETS), required=True)
    ap.add_argument("--train-size", type=int, default=0, help="0 = full training split")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--val-size", type=int, default=3000, help="stratified validation sample used for early stopping")
    ap.add_argument("--eval-every", type=int, default=250, help="steps between validation checks")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(max(torch.get_num_threads(), 8))

    hf_id, family = PRESETS[args.model]
    label_list = labels()
    lid = {l: i for i, l in enumerate(label_list)}
    train, val = load_split("train"), load_split("val")
    if args.train_size and args.train_size < len(train):
        train, _ = train_test_split(train, train_size=args.train_size, stratify=train.label, random_state=args.seed)
    if args.val_size and args.val_size < len(val):
        val, _ = train_test_split(val, train_size=args.val_size, stratify=val.label, random_state=args.seed)

    tok = AutoTokenizer.from_pretrained(hf_id)
    enc = lambda df: dict(tok(df.text.tolist(), truncation=True, max_length=args.max_len,
                              padding="max_length", return_tensors="pt"))
    xtr, ytr = enc(train), torch.tensor(train.label.map(lid).values)
    xva = enc(val)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_id, num_labels=len(label_list), id2label=dict(enumerate(label_list)), label2id=lid)

    steps = args.epochs * ((len(ytr) + args.batch - 1) // args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    best, best_state, step, t0 = -1.0, None, 0, time.time()
    history = []
    with Timer() as t:
        for epoch in range(args.epochs):
            model.train()
            for b, y in batches(xtr, ytr, args.batch, True):
                loss = model(**b, labels=y).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                step += 1
                if step % args.eval_every == 0 or step == steps:
                    f1 = f1_score(val.label.map(lid), predict(model, xva), average="macro")
                    history.append({"step": step, "val_macro_f1": round(f1, 4)})
                    print(f"step {step}/{steps} loss={loss.item():.3f} val_macroF1={f1:.4f} "
                          f"elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
                    if f1 > best:
                        best, best_state = f1, {k: v.detach().clone() for k, v in model.state_dict().items()}
                    model.train()
    model.load_state_dict(best_state)
    scores = {}
    for s in EVAL_SPLITS:
        df = load_split(s)
        scores[s] = evaluate(df.label, [label_list[i] for i in predict(model, enc(df))], label_list)
    save_result(args.model, family,
                {"checkpoint": hf_id, "train_size": len(ytr), "val_size": len(val), "epochs": args.epochs, "batch": args.batch,
                 "lr": args.lr, "max_len": args.max_len, "best_val_macro_f1": round(best, 4),
                 "params": sum(p.numel() for p in model.parameters())},
                scores, t.seconds, {"val_history": history})


if __name__ == "__main__":
    main()
