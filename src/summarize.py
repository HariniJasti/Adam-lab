"""Print every number quoted in README.md straight out of the result JSONs.

Run: python src/summarize.py
"""
import json

import numpy as np

from common import RESULTS


def load(name):
    return json.loads((RESULTS / name).read_text())


def main():
    a = load("e01_adam_by_hand.json")
    print("== e01 ==")
    for q, d in a["worst_abs_diff_per_quantity"].items():
        print(f"  worst |hand-torch| {q:6s} float64={d['f64']:.3e} "
              f"float32={d['f32']:.3e}")
    print("  final w by hand:", a["by_hand"][-1]["w"])
    print("  final w torch64 :", a["torch_float64"][-1]["w"])
    print("  eps-inside-sqrt w:", a["eps_inside_sqrt_final_w"])

    b = load("e02_bias_correction.json")
    print("== e02 ==")
    for k, v in b["analytic"].items():
        print(f"  beta2={k}: peak inflation {v['max_inflation_factor']:.3f}x at step "
              f"{v['argmax_inflation_step']}; within 10/5/1% at "
              f"{v['steps_within_10pct']}/{v['steps_within_5pct']}/"
              f"{v['steps_within_1pct']}")
    print("  r(1), r(20) for beta2=0.999:",
          b["analytic"]["0.999"]["r_first_20"][0],
          b["analytic"]["0.999"]["r_first_20"][19])
    print("  loss@20 bc/nobc:", b["loss_at_step_20"])
    print("  loss@400 bc/nobc:", b["loss_at_final"])
    print("  empirical crossover:", b["empirical_crossover_step"])

    c = load("e02b_tuned_bias_correction.json")
    print("== e02b ==")
    print("  best:", c["best"])
    print("  gap:", abs(c["best"]["bc"]["val"] - c["best"]["nobc"]["val"]))
    for tag in ("bc", "nobc"):
        print(f"  {tag}: " + "  ".join(
            f"{lr}:{c['results'][tag][lr]['val']:.4f}" for lr in c["results"][tag]))

    d = load("e03_update_ratio.json")
    print("== e03 ==")
    print("  warmup:", d["config"]["warmup"], " peak lr:", d["config"]["peak_lr"])
    w = d["warmup100"]
    for g in sorted(w["ratio"]):
        print(f"  {g:9s} ratio@1={w['ratio_at_step1'][g]:.2e} "
              f"ratio@warmup_end={w['ratio_at_end_of_warmup'][g]:.2e} "
              f"ratio(last100)={w['ratio_mean_last100'][g]:.2e} "
              f"(ratio/lr): {w['norm_ratio_at_step1'][g]:.1f} -> "
              f"{w['norm_ratio_last100'][g]:.2f}  plateau@{w['plateau_step'][g]}")
    conv = d["warmup_effect_gone_step"]
    print("  warmup-vs-nowarmup agreement step per group:", conv)
    print("  median:", int(np.median(list(conv.values()))), " max:", max(conv.values()))
    n = d["nowarmup"]
    print("  no-warmup step-1 ratio vs its own last-100 level:")
    for g in sorted(n["ratio"]):
        print(f"    {g:9s} {n['ratio_at_step1'][g]:.2e} -> "
              f"{n['ratio_mean_last100'][g]:.2e}  "
              f"({n['ratio_at_step1'][g]/n['ratio_mean_last100'][g]:.1f}x)")

    e = load("e04_schedules.json")
    print("== e04 ==")
    print("  tuned peak lr:", e["tuned_peak_lr"])
    for k in ("cosine", "wsd"):
        print(f"  {k} sweep:", {lr: round(v["val_300"], 4)
                                for lr, v in e["sweep"][k].items()})
        f = e["final"][k]
        print(f"  {k}: val@200={f['val_200']:.4f} {f['val_200_per_seed']}  "
              f"val@300={f['val_300']:.4f} {f['val_300_per_seed']}")
    print("  wsd forked at 200:", e["wsd_fork_from_200"]["val"])

    f = load("e05_width_lr_sweep.json")
    print("== e05 ==")
    print("  target variance:", f["target_variance"])
    for tag in ("SP", "muP"):
        for w in f["config"]["widths"]:
            r = f["results"][tag][str(w)]
            print(f"  {tag} w={w}: lr*={r['lr_star']:.4g} grid={r['lr_star_grid']:.4g} "
                  f"loss*={r['loss_star']:.5f}")
        print(f"  {tag} extrap:", {k: (round(v, 5) if isinstance(v, float) else v)
                                   for k, v in f["extrapolation"][tag].items()})
    print("  basin (within 10% of best):", json.dumps(f["basin_within_10pct"]))
    cc = f["coordinate_check"]
    for tag in ("SP", "muP"):
        print(f"  coord {tag} l2 delta_rms:",
              {w: round(cc[tag][w]["l2"]["delta_rms"], 4) for w in cc[tag]})

    g = load("e05b_stability_edge.json")
    print("== e05b ==")
    for tag in ("SP", "muP"):
        print(f"  {tag}:", json.dumps(g["edges"][tag]))


if __name__ == "__main__":
    main()
