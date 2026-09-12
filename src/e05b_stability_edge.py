"""Part 5b - the statistic that is actually well identified: the stability edge.

The minimum of a loss-vs-LR curve sits in a flat basin, so its location is
noisy. The right-hand wall -- the smallest LR at which the loss is 2x worse
than the best - is a sharp, monotone feature and is a much better-conditioned
way to see how the usable LR range moves with width.

Reads results/e05_width_lr_sweep.json, writes a figure and a JSON. No training.

Run: python src/e05b_stability_edge.py
"""
from __future__ import annotations

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import FIGURES, RESULTS, save_json

FACTOR = 2.0
BASE = 256
EXTRAP_WIDTH = 4096


def edge(lrs, loss, factor=FACTOR):
    """Log-linear interpolation of where loss first crosses factor * min(loss),
    searching to the right of the grid minimum."""
    v = np.asarray(loss, dtype=float)
    lrs = np.asarray(lrs, dtype=float)
    best = np.nanmin(v)
    thr = factor * best
    i0 = int(np.nanargmin(v))
    for i in range(i0, len(v) - 1):
        a, b = v[i], v[i + 1]
        if not np.isfinite(b) or b >= thr > a:
            if not np.isfinite(b):
                return float(lrs[i + 1])
            f = (thr - a) / (b - a)
            return float(10 ** (np.log10(lrs[i]) + f * (np.log10(lrs[i + 1]) -
                                                        np.log10(lrs[i]))))
    return float(lrs[-1])


def main():
    data = json.loads((RESULTS / "e05_width_lr_sweep.json").read_text())
    widths = data["config"]["widths"]
    out = {}
    for tag in ("SP", "muP"):
        e = {}
        for w in widths:
            r = data["results"][tag][str(w)]
            e[w] = edge(r["lrs"], r["loss"])
        xs = np.log2(np.array(widths) / BASE)
        ys = np.log10([e[w] for w in widths])
        slope, intercept = np.polyfit(xs, ys, 1)
        out[tag] = {
            "edge_lr": {str(w): e[w] for w in widths},
            "slope_log10_per_doubling": float(slope),
            "implied_power_of_width": float(slope / math.log10(2)),
            "pred_edge_at_4096": float(10 ** (intercept + slope *
                                              math.log2(EXTRAP_WIDTH / BASE))),
            "ratio_256_to_1024": float(e[widths[0]] / e[widths[-1]]),
        }
        print(f"{tag}: stability edge "
              f"{'  '.join(f'{w}:{e[w]:.3g}' for w in widths)}  "
              f"-> ~width^{slope/math.log10(2):+.2f}, "
              f"edge(256)/edge(1024) = {out[tag]['ratio_256_to_1024']:.2f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for tag, c, mk in (("SP", "tab:red", "o"), ("muP", "tab:purple", "s")):
        ys = [out[tag]["edge_lr"][str(w)] for w in widths]
        ax.plot(widths, ys, marker=mk, color=c, label=f"{tag} stability edge")
        ys2 = [data["results"][tag][str(w)]["lr_star"] for w in widths]
        ax.plot(widths, ys2, marker=mk, color=c, ls="--", alpha=0.5,
                label=f"{tag} loss minimum")
    ref = np.array(widths, dtype=float)
    ax.plot(ref, out["SP"]["edge_lr"][str(widths[0])] * (ref / widths[0]) ** -1,
            color="k", ls=":", lw=1, label="1/width reference")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("width"); ax.set_ylabel("learning rate")
    ax.set_title(f"usable LR range vs width ({FACTOR:g}x-worse-than-best edge)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "e05b_stability_edge.png", dpi=140)

    coord = coordinate_figure()
    save_json("e05b_stability_edge.json",
              {"factor": FACTOR, "widths": widths, "edges": out,
               "coordinate_check_summary": coord})




def coordinate_figure():
    """Redraw the coordinate check from the saved JSON, showing the deepest
    hidden layer (the one where the SP/muP difference is largest)."""
    data = json.loads((RESULTS / "e05_width_lr_sweep.json").read_text())
    cc = data["coordinate_check"]
    layers = sorted(next(iter(cc["SP"].values())).keys())
    deep = layers[-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, title in zip(axes, ("rms", "delta_rms"),
                              ("RMS of activations after 8 steps",
                               "RMS of the CHANGE in activations over 8 steps")):
        cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(layers)))
        for tag, ls, mk in (("SP", "-", "o"), ("muP", "--", "s")):
            for layer, c in zip(layers, cols):
                ws = sorted(int(w) for w in cc[tag])
                ax.plot(ws, [cc[tag][str(w)][layer][key] for w in ws], ls,
                        marker=mk, color=c, ms=4, label=f"{tag} {layer}")
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("width"); ax.set_ylabel(key)
        ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES / "e05_coordinate_check.png", dpi=140)
    growth = {tag: cc[tag][str(max(int(w) for w in cc[tag]))][deep]["delta_rms"] /
                   cc[tag][str(min(int(w) for w in cc[tag]))][deep]["delta_rms"]
              for tag in ("SP", "muP")}
    print(f"deepest layer ({deep}) delta_rms growth from width "
          f"{min(int(w) for w in cc['SP'])} to {max(int(w) for w in cc['SP'])}: "
          + "  ".join(f"{t}: x{g:.2f}" for t, g in growth.items()))
    return {"layer": deep, "delta_rms_growth_256_to_2048": growth,
            "delta_rms": {t: {w: cc[t][w][deep]["delta_rms"] for w in cc[t]}
                          for t in ("SP", "muP")}}


if __name__ == "__main__":
    main()
