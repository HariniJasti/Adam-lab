"""Part 1 - Adam by hand.

One scalar weight, five prescribed gradients. Compute m, v, m_hat, v_hat and
the update with plain Python floats, then feed the identical gradients to
torch.optim.Adam and compare.

Run: python src/e01_adam_by_hand.py
"""
from __future__ import annotations

import json
import math

import torch

from common import RESULTS, save_json

W0 = 0.7
GRADS = [0.30, -0.10, 0.25, 0.05, -0.40]
LR, B1, B2, EPS = 0.1, 0.9, 0.999, 1e-8


def by_hand():
    """Plain-Python Adam. No tensors, no library calls."""
    w, m, v = W0, 0.0, 0.0
    rows = []
    for t, g in enumerate(GRADS, start=1):
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g * g
        m_hat = m / (1 - B1**t)
        v_hat = v / (1 - B2**t)
        step = LR * m_hat / (math.sqrt(v_hat) + EPS)
        w = w - step
        rows.append(dict(t=t, g=g, m=m, v=v, m_hat=m_hat, v_hat=v_hat,
                         step=step, w=w))
    return rows


def with_pytorch(dtype=torch.float64):
    p = torch.tensor([W0], dtype=dtype, requires_grad=True)
    opt = torch.optim.Adam([p], lr=LR, betas=(B1, B2), eps=EPS)
    out = []
    for g in GRADS:
        prev = p.detach().clone()
        p.grad = torch.tensor([g], dtype=dtype)
        opt.step()
        state = opt.state[p]
        t = int(state["step"].item()) if torch.is_tensor(state["step"]) \
            else int(state["step"])
        m = state["exp_avg"].item()
        v = state["exp_avg_sq"].item()
        out.append(dict(
            t=t,
            m=m,
            v=v,
            # PyTorch never materialises m_hat / v_hat -- it folds the two
            # corrections into step_size and denom. Reconstructed here from its
            # own buffers and its own step counter so all six quantities can be
            # compared, not just the two it stores.
            m_hat=m / (1 - B1**t),
            v_hat=v / (1 - B2**t),
            step=(prev - p.detach()).item(),
            w=p.detach().item(),
        ))
    return out


def main():
    hand = by_hand()
    t64 = with_pytorch(torch.float64)
    t32 = with_pytorch(torch.float32)

    hdr = ("| t | g_t | m_t | v_t | m_hat | v_hat | step | w_t |\n"
           "|---|-----|-----|-----|-------|-------|------|-----|")
    lines = [hdr]
    for r in hand:
        lines.append("| {t} | {g:+.2f} | {m:+.9f} | {v:.9f} | {m_hat:+.9f} | "
                     "{v_hat:.9f} | {step:+.9f} | {w:.9f} |".format(**r))
    table = "\n".join(lines)
    print(table, "\n")

    # Every quantity in the update rule, checked one at a time.
    QUANTS = ["m", "v", "m_hat", "v_hat", "step", "w"]
    cmp_lines = ["| t | " + " | ".join(f"|d {q}|" for q in QUANTS) + " |",
                 "|---|" + "---|" * len(QUANTS)]
    per_quant = {q: {"f64": 0.0, "f32": 0.0} for q in QUANTS}
    for i, (h, a, b) in enumerate(zip(hand, t64, t32), start=1):
        row = []
        for q in QUANTS:
            d64, d32 = abs(h[q] - a[q]), abs(h[q] - b[q])
            per_quant[q]["f64"] = max(per_quant[q]["f64"], d64)
            per_quant[q]["f32"] = max(per_quant[q]["f32"], d32)
            row.append(f"{d64:.2e}")
        cmp_lines.append(f"| {i} | " + " | ".join(row) + " |")
    cmp = "\n".join(cmp_lines) + "\n\n(float64; worst over all 5 steps per quantity)\n\n"
    cmp += "| quantity | worst |hand - torch| float64 | float32 |\n|---|---|---|\n"
    for q in QUANTS:
        cmp += (f"| {q} | {per_quant[q]['f64']:.3e} | "
                f"{per_quant[q]['f32']:.3e} |\n")
    print(cmp)
    worst64 = per_quant["w"]["f64"]
    worst32 = per_quant["w"]["f32"]
    print(f"worst |w| disagreement  float64: {worst64:.3e}   float32: {worst32:.3e}")
    print("worst over ALL quantities float64:",
          f"{max(d['f64'] for d in per_quant.values()):.3e}")

    # Sanity: the first step of Adam is (almost exactly) +/- lr.
    print(f"first step / lr = {hand[0]['step'] / LR:.9f}  (sign-like, magnitude ~1)")

    # Variant check: eps inside the sqrt (Kingma & Ba's 'AdaMax-style' epsilon-hat)
    w, m, v = W0, 0.0, 0.0
    for t, g in enumerate(GRADS, start=1):
        m = B1 * m + (1 - B1) * g
        v = B2 * v + (1 - B2) * g * g
        w -= LR * (m / (1 - B1**t)) / math.sqrt(v / (1 - B2**t) + EPS)
    print(f"eps-inside-sqrt variant final w = {w:.9f} "
          f"(differs from PyTorch by {abs(w - t64[-1]['w']):.3e})")

    save_json("e01_adam_by_hand.json", dict(
        config=dict(w0=W0, grads=GRADS, lr=LR, beta1=B1, beta2=B2, eps=EPS),
        by_hand=hand, torch_float64=t64, torch_float32=t32,
        worst_abs_diff_per_quantity=per_quant,
        worst_abs_diff_w_float64=worst64, worst_abs_diff_w_float32=worst32,
        eps_inside_sqrt_final_w=w,
    ))
    (RESULTS / "e01_table.md").write_text(table + "\n\n" + cmp + "\n")


if __name__ == "__main__":
    main()
