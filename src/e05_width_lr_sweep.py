"""Part 5 - learning-rate sweep at widths 256 / 512 / 1024, SP and muP.

Task: online teacher-student regression (fresh batch every step, so the loss
is a clean function of the optimizer and nothing else). Teacher: 2-hidden-layer
tanh net, 128 -> 512 -> 512 -> 64. Student: MLP 128 -> width x3 -> 64, GELU,
Adam, 300 steps, 2 seeds. The teacher is hard enough that 300 steps leaves the
student far from the noise floor, which is what makes the minimum of the
loss-vs-LR curve identifiable rather than a flat basin.

Two parameterizations, swept over the same LR grid so the x-axes are
comparable (at width 256 = base width they are the same model, and the two
curves must coincide -- that is the sanity check):

  SP  : one LR for every tensor.
  muP : input layer and biases at eta, hidden + readout at eta / m,
        where m = width / 256. No output multiplier (see README).

Run: python src/e05_width_lr_sweep.py
"""
from __future__ import annotations

import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import FIGURES, TeacherStudentData, WidthMLP, save_json, set_seed

WIDTHS = [256, 512, 1024]
BASE = 256
LRS = np.logspace(-4, -1, 10)
DEPTH = 4                 # inp + 3 hidden matrices + readout
STEPS = 300
BATCH = 128
SEEDS = (0, 1)
EXTRAP_WIDTH = 4096


def eval_loss(model, task, n=4096):
    g = torch.Generator().manual_seed(31337)
    x = torch.randn(n, task.d_in, generator=g)
    y = task.f(x)                                   # noiseless targets
    with torch.no_grad():
        pred, _ = model(x)
    return torch.nn.functional.mse_loss(pred, y).item()


def run(width: int, lr: float, mup: bool, seed: int):
    task = TeacherStudentData()
    set_seed(seed)
    model = WidthMLP(width, task.d_in, task.d_out, depth=DEPTH, mup=mup, base=BASE)
    opt = torch.optim.Adam(model.param_groups(lr), betas=(0.9, 0.999))
    g = torch.Generator().manual_seed(2_000 + seed)
    curve = []
    for _ in range(STEPS):
        x, y = task.batch(BATCH, g)
        opt.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if not torch.isfinite(loss):
            return float("nan"), curve
        loss.backward()
        opt.step()
        curve.append(loss.item())
    ev = eval_loss(model, task)
    return (ev if math.isfinite(ev) else float("nan")), curve


def parabola_min(lrs, vals):
    """Refine the grid minimum with a parabola in log10(lr) through the best
    point and its two neighbours. Returns (lr*, loss*)."""
    v = np.asarray(vals, dtype=float)
    i = int(np.nanargmin(v))
    if i == 0 or i == len(v) - 1 or not np.all(np.isfinite(v[i - 1:i + 2])):
        return float(lrs[i]), float(v[i])
    x = np.log10(np.asarray(lrs[i - 1:i + 2], dtype=float))
    y = v[i - 1:i + 2]
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0:
        return float(lrs[i]), float(v[i])
    xs = -b / (2 * a)
    xs = float(np.clip(xs, x[0], x[2]))
    return float(10 ** xs), float(a * xs * xs + b * xs + c)


def coordinate_check(steps=8, lr=3e-3):
    """muP's defining property: the *change* in hidden activations per step is
    width-independent. Measure RMS(h) and RMS(h - h_init) after a few steps."""
    out = {}
    task = TeacherStudentData()
    g_probe = torch.Generator().manual_seed(5)
    x_probe = torch.randn(512, task.d_in, generator=g_probe)
    for mup in (False, True):
        tag = "muP" if mup else "SP"
        out[tag] = {}
        for width in WIDTHS + [2048]:
            set_seed(0)
            model = WidthMLP(width, task.d_in, task.d_out, depth=DEPTH, mup=mup, base=BASE)
            acts = {}

            def grab(name):
                def hook(_m, _i, o):
                    acts[name] = o.detach()
                return hook

            hs = [model.inp.register_forward_hook(grab("l0"))]
            for j, lin in enumerate(model.hidden):
                hs.append(lin.register_forward_hook(grab(f"l{j+1}")))
            with torch.no_grad():
                model(x_probe)
            init = {k: v.clone() for k, v in acts.items()}
            opt = torch.optim.Adam(model.param_groups(lr), betas=(0.9, 0.999))
            g = torch.Generator().manual_seed(11)
            for _ in range(steps):
                xb, yb = task.batch(BATCH, g)
                opt.zero_grad(set_to_none=True)
                _, loss = model(xb, yb)
                loss.backward()
                opt.step()
            with torch.no_grad():
                model(x_probe)
            out[tag][width] = {
                k: {"rms": float(acts[k].pow(2).mean().sqrt()),
                    "delta_rms": float((acts[k] - init[k]).pow(2).mean().sqrt())}
                for k in sorted(acts)
            }
            for h in hs:
                h.remove()
    return out


def main():
    t0 = time.time()
    results = {}
    for mup in (False, True):
        tag = "muP" if mup else "SP"
        results[tag] = {}
        for width in WIDTHS:
            vals = []
            for lr in LRS:
                per_seed = [run(width, float(lr), mup, s)[0] for s in SEEDS]
                vals.append(float(np.nanmean(per_seed)) if np.any(np.isfinite(per_seed))
                            else float("nan"))
            results[tag][width] = {"lrs": LRS.tolist(), "loss": vals}
            lr_star, loss_star = parabola_min(LRS, vals)
            results[tag][width]["lr_star"] = lr_star
            results[tag][width]["loss_star"] = loss_star
            results[tag][width]["lr_star_grid"] = float(LRS[int(np.nanargmin(vals))])
            print(f"{tag} width={width:5d}  lr*={lr_star:.4g} "
                  f"(grid {results[tag][width]['lr_star_grid']:.4g})  "
                  f"loss*={loss_star:.5f}   ({time.time()-t0:.0f}s)")

    # --- extrapolate to width 4096 ---
    extrap = {}
    for tag in ("SP", "muP"):
        xs = np.log2(np.array(WIDTHS) / BASE)
        ys = np.log10([results[tag][w]["lr_star"] for w in WIDTHS])
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = 10 ** (intercept + slope * math.log2(EXTRAP_WIDTH / BASE))
        resid = ys - (intercept + slope * xs)
        extrap[tag] = {
            "slope_log10lr_per_doubling": float(slope),
            "implied_power_of_width": float(slope / math.log10(2)),
            "intercept": float(intercept),
            "pred_lr_at_4096": float(pred),
            "max_abs_residual_dex": float(np.abs(resid).max()),
        }
        print(f"{tag}: d log10(lr*)/d log2(width) = {slope:+.3f} "
              f"(lr* ~ width^{slope/math.log10(2):+.2f}); "
              f"predicted lr* at {EXTRAP_WIDTH} = {pred:.4g}")

    coord = coordinate_check()

    # variance of the (noiseless) targets, i.e. the loss of predicting the mean
    _task = TeacherStudentData()
    _g = torch.Generator().manual_seed(31337)
    target_var = float(_task.f(torch.randn(4096, _task.d_in, generator=_g)).var())
    print(f"target variance (loss of predicting the mean) = {target_var:.4f}")

    # --- curvature of the optimum: how wide is the good basin? ---
    basin = {}
    for tag in ("SP", "muP"):
        basin[tag] = {}
        for w in WIDTHS:
            v = np.array(results[tag][w]["loss"], dtype=float)
            best = np.nanmin(v)
            ok = LRS[np.isfinite(v) & (v <= best * 1.1)]
            basin[tag][w] = {"lr_lo": float(ok.min()), "lr_hi": float(ok.max()),
                             "width_in_dex": float(np.log10(ok.max() / ok.min()))}

    # ---------------- figures ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, tag in zip(axes[:2], ("SP", "muP")):
        for w, c in zip(WIDTHS, ("tab:blue", "tab:orange", "tab:green")):
            v = results[tag][w]["loss"]
            ax.plot(LRS, v, marker="o", ms=4, color=c, label=f"width {w}")
            ax.axvline(results[tag][w]["lr_star"], color=c, ls=":", lw=1)
            ax.scatter([results[tag][w]["lr_star"]], [results[tag][w]["loss_star"]],
                       marker="*", s=220, color=c, edgecolors="k", zorder=6)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("learning rate (eta)"); ax.set_ylabel("eval MSE after 300 steps")
        ax.set_title(f"{tag}: loss vs learning rate"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[2]
    xs = np.array(WIDTHS)
    for tag, c, mk in (("SP", "tab:red", "o"), ("muP", "tab:purple", "s")):
        ys = [results[tag][w]["lr_star"] for w in WIDTHS]
        ax.plot(xs, ys, marker=mk, color=c, label=f"{tag} measured")
        e = extrap[tag]
        grid = np.array([256, 512, 1024, 2048, 4096])
        ax.plot(grid, 10 ** (e["intercept"] + e["slope_log10lr_per_doubling"]
                             * np.log2(grid / BASE)), ls="--", color=c, alpha=0.6)
        ax.scatter([EXTRAP_WIDTH], [e["pred_lr_at_4096"]], marker="*", s=240,
                   color=c, edgecolors="k", zorder=6,
                   label=f"{tag} predicted @4096 = {e['pred_lr_at_4096']:.2g}")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("width"); ax.set_ylabel("optimal learning rate")
    ax.set_title("where the optimum goes"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "e05_width_lr_sweep.png", dpi=140)

    # coordinate-check figure
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in zip(axes2, ("rms", "delta_rms"),
                              ("RMS of hidden activations after 8 steps",
                               "RMS of the CHANGE in activations (8 steps)")):
        for tag, ls in (("SP", "-"), ("muP", "--")):
            for layer, mk in (("l0", "o"), ("l1", "s")):
                ws = sorted(coord[tag])
                ax.plot(ws, [coord[tag][w][layer][key] for w in ws], ls, marker=mk,
                        label=f"{tag} {layer}")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("width"); ax.set_ylabel(key); ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig2.tight_layout()
    fig2.savefig(FIGURES / "e05_coordinate_check.png", dpi=140)

    save_json("e05_width_lr_sweep.json", {
        "config": dict(widths=WIDTHS, base=BASE, lrs=LRS.tolist(), steps=STEPS,
                       batch=BATCH, seeds=list(SEEDS), depth=DEPTH,
                       task="online teacher-student regression",
                       extrap_width=EXTRAP_WIDTH),
        "target_variance": target_var,
        "results": results, "extrapolation": extrap,
        "basin_within_10pct": basin, "coordinate_check": coord,
    })
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
