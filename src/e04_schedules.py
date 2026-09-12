"""Part 4 - cosine vs WSD, both tuned, both stopped at step 200 of 300.

Protocol
  1. Sweep peak LR separately for each schedule -- a schedule comparison where
     only one side is tuned measures the tuning, not the schedule. Selection
     metric: val loss at the end of the 300-step budget (one seed, for CPU
     budget reasons).
  2. Re-run each at its own best LR, snapshotting model + optimizer state at
     step 200.
  3. Report val loss at step 200 and at step 300 for both.
  4. Fork the WSD run at its step-200 snapshot and decay to zero over 30 extra
     steps -- the thing you can actually do with a constant-LR checkpoint and
     cannot do with a half-finished cosine run.

Run: python src/e04_schedules.py
"""
from __future__ import annotations

import copy
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (CharData, FIGURES, TinyTransformer, lr_at, save_json,
                    set_seed, wsd_decay_lr)

TOTAL = 300
STOP_AT = 200
WARMUP = 20
BATCH = 24
SEQ = 128
MIN_FRAC = 0.1
DECAY_FRAC = 0.2          # WSD decays over the last 20% = 60 steps
FORK_DECAY_STEPS = 30
LRS = [1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2]  # grid must bracket both optima
SEEDS = (0, 1)          # seeds for the final head-to-head
TUNE_SEEDS = (0,)       # LR sweep uses one seed to stay inside the CPU budget


def evaluate(model, data, n_batches=16):
    g = torch.Generator().manual_seed(4242)
    model.eval()
    tot = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            x, y = data.batch(BATCH, g, split="val")
            _, loss = model(x, y)
            tot += loss.item()
    model.train()
    return tot / n_batches


def build(seed, data):
    set_seed(seed)
    return TinyTransformer(data.vocab_size, d_model=128, n_layer=3,
                           n_head=4, seq_len=SEQ)


def train(kind: str, peak: float, seed: int, snapshot_at=None):
    data = CharData(seq_len=SEQ)
    model = build(seed, data)
    opt = torch.optim.Adam(model.parameters(), lr=peak, betas=(0.9, 0.999))
    g = torch.Generator().manual_seed(55_000 + seed)
    train_loss, lrs, evals = [], [], {}
    snap = None
    for step in range(TOTAL):
        lr = lr_at(step, total=TOTAL, peak=peak, warmup=WARMUP, kind=kind,
                   min_frac=MIN_FRAC, decay_frac=DECAY_FRAC)
        for grp in opt.param_groups:
            grp["lr"] = lr
        x, y = data.batch(BATCH, g)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        train_loss.append(loss.item())
        lrs.append(lr)
        if snapshot_at is not None and step + 1 == snapshot_at:
            snap = (copy.deepcopy(model.state_dict()),
                    copy.deepcopy(opt.state_dict()),
                    copy.deepcopy(g.get_state()))
            evals[snapshot_at] = evaluate(model, data)
    evals[TOTAL] = evaluate(model, data)
    return dict(train_loss=train_loss, lr=lrs, evals=evals), snap, data


def fork_and_decay(snap, peak: float, seed: int, data, steps=FORK_DECAY_STEPS):
    """Continue from the step-200 snapshot with a short linear decay to zero."""
    model = build(seed, data)
    model.load_state_dict(snap[0])
    opt = torch.optim.Adam(model.parameters(), lr=peak, betas=(0.9, 0.999))
    opt.load_state_dict(snap[1])
    g = torch.Generator()
    g.set_state(snap[2])
    losses, lrs = [], []
    for i in range(steps):
        lr = wsd_decay_lr(i, steps, peak)
        for grp in opt.param_groups:
            grp["lr"] = lr
        x, y = data.batch(BATCH, g)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        lrs.append(lr)
    return dict(train_loss=losses, lr=lrs, val=evaluate(model, data))


def main():
    t0 = time.time()
    sweep = {}
    for kind in ("cosine", "wsd"):
        sweep[kind] = {}
        for peak in LRS:
            vals = []
            for seed in TUNE_SEEDS:
                out, _, _ = train(kind, peak, seed)
                vals.append(out["evals"][TOTAL])
            sweep[kind][f"{peak:g}"] = {"val_300_per_seed": vals,
                                        "val_300": float(np.mean(vals))}
            print(f"{kind:6s} lr={peak:<7g} val@300={np.mean(vals):.4f} "
                  f"{[round(v,4) for v in vals]}  ({time.time()-t0:.0f}s)")

    best = {k: min(v, key=lambda p: v[p]["val_300"]) for k, v in sweep.items()}
    print("\ntuned peak LR:", {k: float(v) for k, v in best.items()})

    final, snaps, datas = {}, {}, {}
    for kind in ("cosine", "wsd"):
        peak = float(best[kind])
        per_seed = []
        for seed in SEEDS:
            out, snap, data = train(kind, peak, seed, snapshot_at=STOP_AT)
            per_seed.append(out)
            if seed == SEEDS[0]:
                snaps[kind], datas[kind] = snap, data
        final[kind] = {
            "peak_lr": peak,
            "val_200_per_seed": [o["evals"][STOP_AT] for o in per_seed],
            "val_300_per_seed": [o["evals"][TOTAL] for o in per_seed],
            "val_200": float(np.mean([o["evals"][STOP_AT] for o in per_seed])),
            "val_300": float(np.mean([o["evals"][TOTAL] for o in per_seed])),
            "train_loss": per_seed[0]["train_loss"],
            "lr": per_seed[0]["lr"],
        }
        print(f"{kind}: val@200={final[kind]['val_200']:.4f}  "
              f"val@300={final[kind]['val_300']:.4f}")

    fork = fork_and_decay(snaps["wsd"], float(best["wsd"]), SEEDS[0], datas["wsd"])
    print(f"WSD forked at 200 and decayed over {FORK_DECAY_STEPS} steps: "
          f"val={fork['val']:.4f}  (cosine@200 = "
          f"{final['cosine']['val_200_per_seed'][0]:.4f}, "
          f"cosine@300 = {final['cosine']['val_300_per_seed'][0]:.4f})")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    ax = axes[0]
    for kind in ("cosine", "wsd"):
        ax.plot(np.arange(1, TOTAL + 1), final[kind]["lr"], label=kind)
    ax.axvline(STOP_AT, color="k", ls="--", lw=1, label="stop at 200")
    ax.set_xlabel("step"); ax.set_ylabel("lr"); ax.set_title("(a) the two schedules")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    k = 15
    for kind in ("cosine", "wsd"):
        tr = np.array(final[kind]["train_loss"])
        ax.plot(np.arange(k, TOTAL + 1), np.convolve(tr, np.ones(k) / k, "valid"),
                label=kind)
    fk = np.array(fork["train_loss"])
    ax.plot(np.arange(STOP_AT + 1, STOP_AT + 1 + len(fk)), fk, lw=1, alpha=0.5,
            color="tab:green")
    ax.plot(np.arange(STOP_AT + k, STOP_AT + 1 + len(fk)),
            np.convolve(fk, np.ones(k) / k, "valid"), color="tab:green",
            label=f"WSD forked at 200, decay {FORK_DECAY_STEPS}")
    ax.axvline(STOP_AT, color="k", ls="--", lw=1)
    ax.set_xlabel("step"); ax.set_ylabel("train loss (15-step mean)")
    ax.set_title("(b) training"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2]
    labels = ["cosine@200", "wsd@200", f"wsd@200+{FORK_DECAY_STEPS} decay",
              "cosine@300", "wsd@300"]
    vals = [final["cosine"]["val_200"], final["wsd"]["val_200"], fork["val"],
            final["cosine"]["val_300"], final["wsd"]["val_300"]]
    cols = ["tab:blue", "tab:orange", "tab:green", "tab:blue", "tab:orange"]
    ax.bar(range(len(vals)), vals, color=cols, alpha=0.85)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(min(vals) - 0.08, max(vals) + 0.05)
    ax.set_ylabel("val loss"); ax.set_title("(c) what you actually get")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES / "e04_schedules.png", dpi=140)

    save_json("e04_schedules.json", {
        "config": dict(total=TOTAL, tune_seeds=list(TUNE_SEEDS), stop_at=STOP_AT, warmup=WARMUP, batch=BATCH,
                       seq=SEQ, min_frac=MIN_FRAC, decay_frac=DECAY_FRAC,
                       lrs=LRS, seeds=list(SEEDS),
                       fork_decay_steps=FORK_DECAY_STEPS),
        "sweep": sweep, "tuned_peak_lr": {k: float(v) for k, v in best.items()},
        "final": final, "wsd_fork_from_200": fork,
    })
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
