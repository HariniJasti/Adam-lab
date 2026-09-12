"""Part 3 - per-layer update-to-weight ratio under warmup.

For every parameter tensor and every step we log

    ratio_l(t) = RMS(w_l(t+1) - w_l(t)) / RMS(w_l(t))

which is the scale-free "how far did this layer move relative to its own size"
number. Two runs: 100-step linear warmup then constant LR, and no warmup at all.

"The step at which warmup stops changing it" is answered two ways:
  1. mechanically: the LR stops moving at the end of warmup, step W.
  2. empirically: normalise by the LR actually used, ratio_l(t) / lr(t), and
     find the first step after which that quantity stays within +/-10% of its
     late-training plateau. If warmup were the only thing driving the ratio,
     the normalised curve would be flat from step 1.

Run: python src/e03_update_ratio.py
"""
from __future__ import annotations

import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import (CharData, FIGURES, TinyTransformer, lr_at, save_json,
                    set_seed)

STEPS = 400
WARMUP = 100
PEAK_LR = 1e-3
BATCH = 24
SEQ = 128


def group_of(name: str) -> str:
    """Layer group, or None for tensors the ratio is meaningless for.

    Biases (and anything else initialised to exactly zero) are excluded: the
    denominator RMS(w) starts at 0, so the ratio is undefined for them.
    """
    if name.endswith("bias") and "ln" not in name:
        return None
    if "ln" in name and name.endswith("bias"):
        return None
    if name.startswith("tok"):
        return "tok_emb"
    if name.startswith("pos"):
        return "pos_emb"
    if name.startswith("head"):
        return "head"
    if "ln" in name:
        return "ln_gain"
    if "attn.in_proj_weight" in name:
        return "attn_qkv"
    if "attn.in_proj_bias" in name:
        return None
    if "attn.out_proj.weight" in name:
        return "attn_out"
    if "fc1.weight" in name:
        return "mlp_fc1"
    if "fc2.weight" in name:
        return "mlp_fc2"
    return None


def run(warmup: int, seed: int = 0):
    set_seed(seed)
    data = CharData(seq_len=SEQ)
    model = TinyTransformer(data.vocab_size, d_model=128, n_layer=3,
                            n_head=4, seq_len=SEQ)
    named = [(n, p) for n, p in model.named_parameters()]
    opt = torch.optim.Adam(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.999))
    g = torch.Generator().manual_seed(777 + seed)
    hist = {n: [] for n, _ in named}
    lrs, losses = [], []
    for step in range(STEPS):
        lr = lr_at(step, total=STEPS, peak=PEAK_LR, warmup=warmup, kind="constant")
        for grp in opt.param_groups:
            grp["lr"] = lr
        x, y = data.batch(BATCH, g)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        before = {n: p.detach().clone() for n, p in named}
        opt.step()
        for n, p in named:
            w = before[n]
            upd = (p.detach() - w)
            r = upd.pow(2).mean().sqrt().item() / max(w.pow(2).mean().sqrt().item(), 1e-12)
            hist[n].append(r)
        lrs.append(lr)
        losses.append(loss.item())
    return hist, lrs, losses


def aggregate(hist):
    """Average the per-tensor ratios into layer groups (weight-like tensors)."""
    groups = {}
    for n, series in hist.items():
        g = group_of(n)
        if g is None:
            continue
        groups.setdefault(g, []).append(np.array(series))
    return {g: np.mean(np.stack(v), axis=0) for g, v in groups.items()}


def smooth(x: np.ndarray, w: int = 21) -> np.ndarray:
    """Centred moving average, edge-padded, so index t still means step t."""
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(w) / w, mode="valid")


def plateau_step(norm: np.ndarray, tol: float = 0.10, tail: int = 100,
                 w: int = 21) -> int:
    """First step after which the smoothed curve stays within +/-tol of its
    late-training level. Smoothing first, because minibatch noise alone puts
    the raw per-step ratio outside any +/-10% band forever."""
    s = smooth(norm, w)
    ref = s[-tail:].mean()
    ok = np.abs(s - ref) <= tol * ref
    for i in range(len(ok)):
        if ok[i:].all():
            return i + 1  # 1-indexed step
    return len(ok)


def main():
    t0 = time.time()
    out = {}
    for tag, wu in (("warmup100", WARMUP), ("nowarmup", 0)):
        hist, lrs, losses = run(wu)
        groups = aggregate(hist)
        lrs = np.array(lrs)
        norm = {g: v / lrs for g, v in groups.items()}
        out[tag] = {
            "warmup": wu,
            "lr": lrs.tolist(),
            "loss": losses,
            "ratio": {g: v.tolist() for g, v in groups.items()},
            "ratio_over_lr": {g: v.tolist() for g, v in norm.items()},
            "plateau_step": {g: int(plateau_step(v)) for g, v in norm.items()},
            "ratio_at_step1": {g: float(v[0]) for g, v in groups.items()},
            "ratio_at_end_of_warmup": {g: float(v[max(wu, 1) - 1]) for g, v in groups.items()},
            "ratio_mean_last100": {g: float(v[-100:].mean()) for g, v in groups.items()},
            "norm_ratio_at_step1": {g: float(v[0]) for g, v in norm.items()},
            "norm_ratio_last100": {g: float(v[-100:].mean()) for g, v in norm.items()},
            "per_tensor_ratio": {n: v for n, v in hist.items()},
            "per_tensor_group": {n: group_of(n) for n in hist},
        }
        print(f"[{tag}] {time.time()-t0:.0f}s  final loss {losses[-1]:.3f}")
        for g in sorted(groups):
            o = out[tag]
            print(f"   {g:9s} plateau@{o['plateau_step'][g]:4d}  "
                  f"ratio(last100)={o['ratio_mean_last100'][g]:.2e}  "
                  f"(ratio/lr)@1={o['norm_ratio_at_step1'][g]:.2f} -> "
                  f"{o['norm_ratio_last100'][g]:.2f}")

    # ---- the cleanest definition: when do the two runs stop differing? ----
    # Both runs see identical batches from identical init, so the difference in
    # ratio/lr between them is caused by warmup and nothing else.
    conv, rel_curves = {}, {}
    for g in out["warmup100"]["ratio_over_lr"]:
        a = smooth(np.array(out["warmup100"]["ratio_over_lr"][g]))
        b = smooth(np.array(out["nowarmup"]["ratio_over_lr"][g]))
        rel = np.abs(a - b) / np.maximum(b, 1e-12)
        rel_curves[g] = rel
        step = len(rel)
        for i in range(len(rel)):
            if np.all(rel[i:] <= 0.10):
                step = i + 1
                break
        conv[g] = int(step)
    print("\nwarmup vs no-warmup, same batches: ratio/lr agrees to within 10% from")
    for g in sorted(conv):
        print(f"   {g:9s} step {conv[g]}")
    print(f"   -> slowest layer: step {max(conv.values())}")

    # ---- figure ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for col, tag in enumerate(("warmup100", "nowarmup")):
        d = out[tag]
        steps = np.arange(1, STEPS + 1)
        ax = axes[0][col]
        for g, v in sorted(d["ratio"].items()):
            ax.plot(steps, v, lw=1.2, label=g)
        if d["warmup"]:
            ax.axvline(d["warmup"], color="k", ls="--", lw=1, label="warmup ends")
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_title(f"RMS(update)/RMS(weight) - {tag}")
        ax.set_xlabel("step"); ax.set_ylabel("ratio")
        ax.grid(alpha=0.3)
        if col == 0:
            ax.legend(fontsize=7, ncol=2)

        ax = axes[1][col]
        for g, v in sorted(d["ratio_over_lr"].items()):
            ax.plot(steps, v, lw=1.2, label=g)
        if d["warmup"]:
            ax.axvline(d["warmup"], color="k", ls="--", lw=1)
        mx = max(int(s) for s in d["plateau_step"].values())
        ax.axvline(mx, color="r", ls=":", lw=1.2,
                   label=f"last layer plateaus @ {mx}")
        ax.set_yscale("log"); ax.set_xscale("log")
        ax.set_title(f"ratio / lr(t) - {tag}")
        ax.set_xlabel("step"); ax.set_ylabel("ratio / lr")
        ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "e03_update_ratio.png", dpi=140)

    fig2, ax = plt.subplots(figsize=(7.5, 4.6))
    for g, rel in sorted(rel_curves.items()):
        ax.plot(np.arange(1, STEPS + 1), rel, lw=1.2, label=f"{g} (step {conv[g]})")
    ax.axhline(0.10, color="k", ls="-", lw=0.8)
    ax.axvline(WARMUP, color="k", ls="--", lw=1, label="warmup ends (100)")
    ax.axvline(max(conv.values()), color="r", ls=":", lw=1.4,
               label=f"last layer converges ({max(conv.values())})")
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("|ratio/lr difference| (relative)")
    ax.set_title("warmup vs no warmup, identical batches")
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig2.tight_layout()
    fig2.savefig(FIGURES / "e03_warmup_convergence.png", dpi=140)

    save_json("e03_update_ratio.json", {
        "config": dict(peak_lr=PEAK_LR, warmup=WARMUP, steps=STEPS,
                       batch=BATCH, seq=SEQ, schedule="warmup then constant"),
        "warmup_effect_gone_step": conv,
        "warmup_effect_gone_step_max": int(max(conv.values())),
        **out,
    })
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
