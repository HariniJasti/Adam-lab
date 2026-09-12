"""Part 2 - bias correction on vs off.

(a) Closed form. With a stationary gradient the corrected and uncorrected
    updates differ by exactly  r(t) = sqrt(1 - b2^t) / (1 - b1^t)  --
    the uncorrected step is 1/r(t) times too large.
(b) Empirically, on the tiny transformer, 2 seeds each, constant LR, no warmup,
    identical batches, first 20 steps and then 400.

Run: python src/e02_bias_correction.py
"""
from __future__ import annotations

import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (CharData, FIGURES, MyAdam, TinyTransformer, save_json,
                    set_seed)

B1 = 0.9
STEPS = 400
LR = 1e-3
BATCH = 24
SEQ = 128


def ratio(t: int, b1: float, b2: float) -> float:
    """corrected_step / uncorrected_step for a stationary gradient."""
    return math.sqrt(1 - b2**t) / (1 - b1**t)


def steps_until_within(tol: float, b1: float, b2: float, cap: int = 200_000) -> int:
    for t in range(1, cap + 1):
        if all(abs(ratio(s, b1, b2) - 1) <= tol for s in range(t, min(t + 50, cap))):
            return t
    return -1


def analytic():
    out = {}
    for b2 in (0.999, 0.99, 0.95):
        out[str(b2)] = {
            "r_first_20": [ratio(t, B1, b2) for t in range(1, 21)],
            "max_inflation_factor": max(1 / ratio(t, B1, b2) for t in range(1, 21)),
            "argmax_inflation_step": int(np.argmax(
                [1 / ratio(t, B1, b2) for t in range(1, 21)]) + 1),
            "steps_within_10pct": steps_until_within(0.10, B1, b2),
            "steps_within_5pct": steps_until_within(0.05, B1, b2),
            "steps_within_1pct": steps_until_within(0.01, B1, b2),
        }
    return out


def train(bias_correction: bool, seed: int, steps: int = STEPS):
    set_seed(seed)
    data = CharData(seq_len=SEQ)
    model = TinyTransformer(data.vocab_size, d_model=128, n_layer=3,
                            n_head=4, seq_len=SEQ)
    opt = MyAdam(model.parameters(), lr=LR, betas=(B1, 0.999), eps=1e-8,
                 bias_correction=bias_correction)
    g = torch.Generator().manual_seed(10_000 + seed)  # identical batches both ways
    losses, upd_rms = [], []
    for _ in range(steps):
        x, y = data.batch(BATCH, g)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        before = [p.detach().clone() for p in model.parameters()]
        opt.step()
        d2 = sum(((p.detach() - b) ** 2).sum().item()
                 for p, b in zip(model.parameters(), before))
        n = sum(p.numel() for p in model.parameters())
        upd_rms.append(math.sqrt(d2 / n))
        losses.append(loss.item())
    return losses, upd_rms


def main():
    t0 = time.time()
    ana = analytic()
    for b2, d in ana.items():
        print(f"beta2={b2}: uncorrected step is up to {d['max_inflation_factor']:.2f}x "
              f"too large (worst at step {d['argmax_inflation_step']}); "
              f"within 10%/5%/1% of corrected after "
              f"{d['steps_within_10pct']}/{d['steps_within_5pct']}/"
              f"{d['steps_within_1pct']} steps")

    runs = {}
    for bc in (True, False):
        for seed in (0, 1):
            key = f"{'bc' if bc else 'nobc'}_s{seed}"
            losses, upd = train(bc, seed)
            runs[key] = {"loss": losses, "update_rms": upd}
            print(f"  {key}: loss[0]={losses[0]:.3f} loss[19]={losses[19]:.3f} "
                  f"loss[-1]={losses[-1]:.3f}  ({time.time()-t0:.0f}s)")

    bc = np.array([runs["bc_s0"]["loss"], runs["bc_s1"]["loss"]])
    nb = np.array([runs["nobc_s0"]["loss"], runs["nobc_s1"]["loss"]])
    # Smooth to separate signal from minibatch noise.
    k = 20
    def smooth(a):
        return np.array([np.convolve(x, np.ones(k) / k, mode="valid") for x in a])
    bcs, nbs = smooth(bc), smooth(nb)
    gap = np.abs(bcs.mean(0) - nbs.mean(0))
    seed_noise = np.abs(bcs[0] - bcs[1])  # same-config seed-to-seed spread
    crossover = next((i + k for i in range(len(gap))
                      if np.all(gap[i:] <= seed_noise[i:])), None)
    print(f"\nsmoothed |bc - nobc| loss gap falls inside the seed-to-seed spread "
          f"from step {crossover}")

    # ---- figures ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    ax = axes[0]
    ts = np.arange(1, 21)
    for b2, style in zip((0.999, 0.99, 0.95), ("-", "--", ":")):
        ax.plot(ts, [1 / ratio(t, B1, b2) for t in ts], style, marker="o", ms=3,
                label=f"beta2={b2}")
    ax.axhline(1.0, color="k", lw=0.8)
    ax.set_xlabel("step"); ax.set_ylabel("uncorrected step / corrected step")
    ax.set_title("(a) how much too large the uncorrected step is")
    ax.set_yscale("log"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(range(1, 21), bc.mean(0)[:20], marker="o", ms=3, label="bias correction ON")
    ax.plot(range(1, 21), nb.mean(0)[:20], marker="s", ms=3, label="bias correction OFF")
    ax.set_xlabel("step"); ax.set_ylabel("training loss")
    ax.set_title("(b) first 20 steps (transformer, mean of 2 seeds)")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    x = np.arange(k, k + len(gap))
    ax.plot(x, bcs.mean(0), label="bias correction ON")
    ax.plot(x, nbs.mean(0), label="bias correction OFF")
    ax.fill_between(x, bcs.min(0), bcs.max(0), alpha=0.2)
    ax.fill_between(x, nbs.min(0), nbs.max(0), alpha=0.2)
    if crossover:
        ax.axvline(crossover, color="k", ls="--", lw=1,
                   label=f"gap < seed noise from step {crossover}")
    ax.set_xlabel("step"); ax.set_ylabel(f"training loss ({k}-step mean)")
    ax.set_title("(c) the whole run"); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES / "e02_bias_correction.png", dpi=140)

    save_json("e02_bias_correction.json", {
        "config": dict(lr=LR, betas=[B1, 0.999], steps=STEPS, batch=BATCH,
                       seq=SEQ, warmup=0, schedule="constant"),
        "analytic": ana,
        "runs": runs,
        "empirical_crossover_step": crossover,
        "loss_at_step_20": {"bc": float(bc.mean(0)[19]), "nobc": float(nb.mean(0)[19])},
        "loss_at_final": {"bc": float(bc.mean(0)[-1]), "nobc": float(nb.mean(0)[-1])},
    })
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
