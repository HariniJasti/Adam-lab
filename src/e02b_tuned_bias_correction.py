"""Part 2b - the comparison bias correction deserves: tune both sides.

Turning bias correction off does not leave the optimizer alone, it multiplies
the early step sizes by 1/r(t). So an untuned A/B is really a comparison of two
different step-size schedules. Here both sides get their own LR sweep and only
the best-of-each are compared.

Run: python src/e02b_tuned_bias_correction.py
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (CharData, FIGURES, MyAdam, TinyTransformer, save_json,
                    set_seed)

STEPS = 250
BATCH = 24
SEQ = 128
LRS = [2e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3]


def eval_loss(model, data, n_batches=12):
    g = torch.Generator().manual_seed(999)
    tot = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = data.batch(BATCH, g, split="val")
            _, loss = model(x, y)
            tot += loss.item()
    return tot / n_batches


def train(bias_correction: bool, lr: float, seed: int = 0):
    set_seed(seed)
    data = CharData(seq_len=SEQ)
    model = TinyTransformer(data.vocab_size, d_model=128, n_layer=3,
                            n_head=4, seq_len=SEQ)
    opt = MyAdam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8,
                 bias_correction=bias_correction)
    g = torch.Generator().manual_seed(10_000 + seed)
    losses = []
    for _ in range(STEPS):
        x, y = data.batch(BATCH, g)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, eval_loss(model, data)


def main():
    t0 = time.time()
    res = {}
    for bc in (True, False):
        tag = "bc" if bc else "nobc"
        res[tag] = {}
        for lr in LRS:
            losses, ev = train(bc, lr)
            res[tag][f"{lr:g}"] = {"train": losses, "val": ev}
            print(f"{tag} lr={lr:<8g} val={ev:.4f}  ({time.time()-t0:.0f}s)")

    best = {}
    for tag, d in res.items():
        finite = {k: v for k, v in d.items() if np.isfinite(v["val"])}
        k = min(finite, key=lambda k: finite[k]["val"])
        best[tag] = {"lr": float(k), "val": finite[k]["val"]}
    print("\nbest per side:", best)
    print(f"best-of-each gap in val loss: "
          f"{abs(best['bc']['val'] - best['nobc']['val']):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for tag, mk in (("bc", "o"), ("nobc", "s")):
        vals = [res[tag][f"{lr:g}"]["val"] for lr in LRS]
        ax.plot(LRS, vals, marker=mk, label=f"bias correction {'ON' if tag=='bc' else 'OFF'}")
        i = int(np.nanargmin(np.where(np.isfinite(vals), vals, np.nan)))
        ax.scatter([LRS[i]], [vals[i]], s=140, facecolors="none",
                   edgecolors="k", zorder=5)
    ax.set_xscale("log"); ax.set_xlabel("peak learning rate")
    ax.set_ylabel(f"val loss after {STEPS} steps")
    ax.set_title("each side tuned separately"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    for tag in ("bc", "nobc"):
        lr = best[tag]["lr"]
        tr = np.array(res[tag][f"{lr:g}"]["train"])
        k = 15
        ax.plot(np.arange(k, len(tr) + 1), np.convolve(tr, np.ones(k) / k, "valid"),
                label=f"{'ON' if tag=='bc' else 'OFF'} @ lr={lr:g}")
    ax.set_xlabel("step"); ax.set_ylabel("train loss (15-step mean)")
    ax.set_title("best vs best"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "e02b_tuned_bias_correction.png", dpi=140)

    save_json("e02b_tuned_bias_correction.json",
              {"config": dict(steps=STEPS, batch=BATCH, seq=SEQ, lrs=LRS),
               "results": res, "best": best})
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
