"""MyAdam must equal torch.optim.Adam when bias correction is on.

Run: python src/test_myadam.py
"""
import sys

import torch

from common import MyAdam, TinyTransformer, CharData, set_seed


def test_scalar_and_vector():
    worst = {}
    for dtype in (torch.float64, torch.float32):
        torch.manual_seed(0)
        a = torch.randn(7, dtype=dtype, requires_grad=True)
        b = a.detach().clone().requires_grad_(True)
        o1 = torch.optim.Adam([a], lr=3e-3, betas=(0.9, 0.999), eps=1e-8)
        o2 = MyAdam([b], lr=3e-3, betas=(0.9, 0.999), eps=1e-8)
        g = torch.randn(20, 7, dtype=dtype)
        w = 0.0
        for i in range(20):
            a.grad, b.grad = g[i].clone(), g[i].clone()
            o1.step()
            o2.step()
            w = max(w, (a - b).abs().max().item())
        worst[str(dtype)] = w
    return worst


def test_transformer(dtype=torch.float64):
    """Same check on every parameter of a real model for 10 steps."""
    set_seed(0)
    data = CharData(seq_len=64)
    ref = TinyTransformer(data.vocab_size, d_model=64, n_layer=2, seq_len=64).to(dtype)
    mine = TinyTransformer(data.vocab_size, d_model=64, n_layer=2, seq_len=64).to(dtype)
    mine.load_state_dict(ref.state_dict())
    o1 = torch.optim.Adam(ref.parameters(), lr=1e-3)
    o2 = MyAdam(mine.parameters(), lr=1e-3)
    g = torch.Generator().manual_seed(7)
    worst = 0.0
    for _ in range(10):
        x, y = data.batch(16, g)
        for model, opt in ((ref, o1), (mine, o2)):
            opt.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            opt.step()
        for pa, pb in zip(ref.parameters(), mine.parameters()):
            worst = max(worst, (pa - pb).abs().max().item())
    return worst


if __name__ == "__main__":
    w = test_scalar_and_vector()
    t64 = test_transformer(torch.float64)
    t32 = test_transformer(torch.float32)
    print("scalar/vector, 20 steps, max |diff|:", w)
    print("transformer (all params), 10 steps, float64 max |diff|:", t64)
    print("transformer (all params), 10 steps, float32 max |diff|:", t32)
    print("  (the float32 gap is accumulated rounding: torch.optim.Adam folds the")
    print("   bias corrections into step_size and denom, MyAdam divides m and v")
    print("   directly. Same algebra, different rounding, compounded over 10 steps.)")
    ok = (w["torch.float64"] < 1e-15 and w["torch.float32"] == 0.0
          and t64 < 1e-12 and t32 < 1e-3)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
