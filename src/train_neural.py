"""Neural baselines from the course (Week 3), trained from scratch (random embeddings):

* TextCNN  (Kim, 2014): parallel filters of width 2/3/4/5 over word embeddings + max pooling
* BiLSTM   : 1-layer bidirectional LSTM, last forward and backward states concatenated

Both use a word vocabulary built from the training split only. Training uses
early stopping on validation macro-F1. Each model is trained with several seeds and the
mean/std is reported, because small neural models vary from run to run.
"""
import argparse
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from common import EVAL_SPLITS, Timer, evaluate, labels, load_split, save_result, seed_everything

TOKEN_RE = re.compile(r"[a-z0-9]+|[^\sa-z0-9]")


def tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall(s.lower())


class Vocab:
    def __init__(self, texts, min_freq=2):
        c = Counter(t for s in texts for t in tokenize(s))
        self.itos = ["<pad>", "<unk>"] + [w for w, n in c.most_common() if n >= min_freq]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, texts, max_len):
        x = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, s in enumerate(texts):
            ids = [self.stoi.get(t, 1) for t in tokenize(s)][:max_len]
            x[i, :len(ids)] = ids
        return torch.from_numpy(x)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes, emb=128, filters=100, widths=(2, 3, 4, 5), dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(emb, filters, w, padding=w - 1) for w in widths)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(filters * len(widths), n_classes)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)                       # (batch, emb, seq)
        pooled = [torch.relu(c(e)).max(dim=2).values for c in self.convs]  # max pooling over time
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, n_classes, emb=128, hidden=128, dropout=0.5):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden, n_classes)            # forward + backward = 2 * hidden

    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(self.emb(x), lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.fc(self.drop(torch.cat([h[-2], h[-1]], dim=1)))


ARCHS = {"textcnn": TextCNN, "bilstm": BiLSTM}


def predict(model, x, bs=512):
    model.eval()
    with torch.no_grad():
        return torch.cat([model(x[i:i + bs]).argmax(1) for i in range(0, len(x), bs)]).numpy()


def run(arch, seed, data, args):
    seed_everything(seed)
    vocab, xtr, ytr, xva, yva, n_classes = data
    model = ARCHS[arch](len(vocab.itos), n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, bad = -1.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(xtr))
        for i in range(0, len(xtr), args.batch):
            idx = perm[i:i + args.batch]
            opt.zero_grad()
            loss_fn(model(xtr[idx]), ytr[idx]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # gradient clipping (Week 3 training tips)
            opt.step()
        f1 = f1_score(yva, predict(model, xva), average="macro")
        print(f"  {arch} seed={seed} epoch={epoch + 1} val_macroF1={f1:.4f}")
        if f1 > best:
            best, best_state, bad = f1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= args.patience:                            # early stopping
                break
    model.load_state_dict(best_state)
    return model, best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", nargs="*", default=list(ARCHS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[42, 43, 44])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=40)
    args = ap.parse_args()
    torch.set_num_threads(max(torch.get_num_threads(), 8))

    label_list = labels()
    lid = {l: i for i, l in enumerate(label_list)}
    train, val = load_split("train"), load_split("val")
    vocab = Vocab(train.text)
    enc = lambda df: (vocab.encode(df.text.tolist(), args.max_len), torch.tensor(df.label.map(lid).values))
    xtr, ytr = enc(train)
    xva, yva = enc(val)
    data = (vocab, xtr, ytr, xva, yva.numpy(), len(label_list))
    tests = {s: enc(load_split(s)) for s in EVAL_SPLITS}

    for arch in args.archs:
        runs = []
        with Timer() as t:
            for seed in args.seeds:
                model, val_f1 = run(arch, seed, data, args)
                runs.append((val_f1, {s: evaluate([label_list[i] for i in y.numpy()],
                                                  [label_list[i] for i in predict(model, x)], label_list)
                                      for s, (x, y) in tests.items()}))
        # report the seed with the median validation score, plus mean/std over seeds
        runs.sort(key=lambda r: r[0])
        median_scores = runs[len(runs) // 2][1]
        seed_stats = {s: {"macro_f1_mean": float(np.mean([r[1][s]["macro_f1"] for r in runs])),
                          "macro_f1_std": float(np.std([r[1][s]["macro_f1"] for r in runs]))} for s in EVAL_SPLITS}
        save_result(arch, "baseline (neural, from scratch)",
                    {"vocab_size": len(vocab.itos), "max_len": args.max_len, "epochs_max": args.epochs,
                     "lr": args.lr, "batch": args.batch, "seeds": args.seeds,
                     "params": sum(p.numel() for p in ARCHS[arch](len(vocab.itos), len(label_list)).parameters())},
                    median_scores, t.seconds / len(args.seeds), {"seed_stats": seed_stats})


if __name__ == "__main__":
    main()
