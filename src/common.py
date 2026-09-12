"""Shared pieces: data, models, LR schedules, muP parameter groups.

Everything here is deterministic given a seed and runs on CPU.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("LAB_THREADS", "2")))

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_json(name: str, obj) -> pathlib.Path:
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2))
    return p


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
_STDLIB = pathlib.Path("/usr/lib/python3.12")


def build_corpus(target_bytes: int = 1_500_000) -> str:
    """Deterministic char corpus: the first N bytes of sorted stdlib sources.

    Real, structured text; no network access; byte-identical on every run.
    """
    cache = RESULTS / "corpus.txt"
    if cache.exists():
        return cache.read_text()
    chunks, total = [], 0
    for path in sorted(_STDLIB.glob("*.py")):
        try:
            txt = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        chunks.append(txt)
        total += len(txt)
        if total >= target_bytes:
            break
    corpus = "".join(chunks)[:target_bytes]
    cache.write_text(corpus)
    return corpus


class CharData:
    def __init__(self, seq_len: int, val_frac: float = 0.1):
        corpus = build_corpus()
        vocab = sorted(set(corpus))
        self.stoi = {c: i for i, c in enumerate(vocab)}
        self.vocab_size = len(vocab)
        ids = torch.tensor([self.stoi[c] for c in corpus], dtype=torch.long)
        n_val = int(len(ids) * val_frac)
        self.train, self.val = ids[:-n_val], ids[-n_val:]
        self.seq_len = seq_len

    def batch(self, bs: int, gen: torch.Generator, split: str = "train"):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - self.seq_len - 1, (bs,), generator=gen)
        x = torch.stack([data[i : i + self.seq_len] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + self.seq_len] for i in ix])
        return x, y


class TeacherStudentData:
    """Online synthetic regression: fresh data every step, so no overfitting
    and the loss after N steps measures optimization only.

    The teacher is a 2-hidden-layer tanh network, deliberately hard enough that
    300 steps leaves the student well short of the noise floor -- otherwise the
    loss-vs-LR curve is flat and its minimum is not identifiable.
    """

    def __init__(self, d_in: int = 128, d_out: int = 64, hidden: int = 512,
                 teacher_depth: int = 2, noise: float = 0.02, seed: int = 1234):
        g = torch.Generator().manual_seed(seed)
        self.ws = [torch.randn(d_in, hidden, generator=g) / math.sqrt(d_in)]
        for _ in range(teacher_depth - 1):
            self.ws.append(torch.randn(hidden, hidden, generator=g) / math.sqrt(hidden))
        self.wo = torch.randn(hidden, d_out, generator=g) / math.sqrt(hidden)
        self.d_in, self.d_out, self.noise = d_in, d_out, noise

    def f(self, x):
        h = x
        for w in self.ws:
            h = torch.tanh(h @ w)
        return h @ self.wo

    def batch(self, bs: int, gen: torch.Generator):
        x = torch.randn(bs, self.d_in, generator=gen)
        y = self.f(x) + self.noise * torch.randn(bs, self.d_out, generator=gen)
        return x, y


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True, bias=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        h = self.ln2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


class TinyTransformer(nn.Module):
    """Small pre-LN decoder. Standard parameterization."""

    def __init__(self, vocab: int, d_model: int = 128, n_layer: int = 3,
                 n_head: int = 4, seq_len: int = 128):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.seq_len = seq_len
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, idx, targets=None):
        b, t = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))
        m = self.mask[:t, :t]
        for blk in self.blocks:
            x = blk(x, m)
        logits = self.head(self.ln_f(x))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class WidthMLP(nn.Module):
    """Depth-fixed MLP whose width we sweep. Supports SP and muP.

    muP (base width `base`, multiplier m = width / base), relative to SP:
      - input layer and all biases: lr = eta          (unchanged)
      - hidden layers:              lr = eta / m
      - readout:                    lr = eta / m
    Initialisation is std = 1/sqrt(fan_in) everywhere in both schemes, which is
    already the muP-correct scaling for the hidden and readout matrices, so the
    two parameterizations differ only in the per-tensor learning rates and are
    numerically identical at width == base.

    No extra output multiplier is used: with init std = 1/sqrt(fan_in) the
    readout already produces Theta(1) logits, and an additional 1/m factor
    would force the deep activations to grow with width to compensate. The
    coordinate check in e05 is what caught that (see README).
    """

    def __init__(self, width: int, d_in: int, d_out: int, depth: int = 3,
                 mup: bool = False, base: int = 256):
        super().__init__()
        self.mup = mup
        self.m = width / base
        self.inp = nn.Linear(d_in, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth - 1)])
        self.out = nn.Linear(width, d_out)
        for lin in [self.inp, *self.hidden, self.out]:
            fan_in = lin.weight.shape[1]
            nn.init.normal_(lin.weight, std=1.0 / math.sqrt(fan_in))
            nn.init.zeros_(lin.bias)

    def forward(self, x, targets=None):
        h = F.gelu(self.inp(x))
        for lin in self.hidden:
            h = F.gelu(lin(h))
        y = self.out(h)
        if targets is None:
            return y, None
        return y, F.mse_loss(y, targets)

    def param_groups(self, lr: float):
        if not self.mup:
            return [{"params": list(self.parameters()), "lr": lr, "name": "all"}]
        wide, narrow = [], []
        for lin in [*self.hidden, self.out]:
            wide.append(lin.weight)
        narrow += [self.inp.weight, self.inp.bias]
        narrow += [lin.bias for lin in [*self.hidden, self.out]]
        return [
            {"params": narrow, "lr": lr, "name": "width-independent"},
            {"params": wide, "lr": lr / self.m, "name": "width-scaled"},
        ]


# --------------------------------------------------------------------------
# Learning-rate schedules
# --------------------------------------------------------------------------
def lr_at(step: int, *, total: int, peak: float, warmup: int, kind: str,
          min_frac: float = 0.1, decay_frac: float = 0.2) -> float:
    """step is 0-indexed: the LR used for the update that produces step+1."""
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    if kind == "constant":
        return peak
    if kind == "cosine":
        prog = (step - warmup) / max(1, total - warmup)
        prog = min(max(prog, 0.0), 1.0)
        return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * prog)))
    if kind == "wsd":
        decay_steps = max(1, int(round(decay_frac * total)))
        start = total - decay_steps
        if step < start:
            return peak
        prog = (step - start) / decay_steps
        prog = min(max(prog, 0.0), 1.0)
        return peak * (1 - prog * (1 - min_frac))  # linear decay to min_frac * peak
    raise ValueError(kind)


def wsd_decay_lr(step_in_decay: int, decay_steps: int, peak: float) -> float:
    """Linear-to-zero decay branch used when forking a WSD run mid-training."""
    prog = min(max(step_in_decay / decay_steps, 0.0), 1.0)
    return peak * (1 - prog)


# --------------------------------------------------------------------------
# A transparent Adam, with bias correction as a switch
# --------------------------------------------------------------------------
class MyAdam(torch.optim.Optimizer):
    """Adam written out longhand so bias correction can be switched off.

    With bias_correction=True this reproduces torch.optim.Adam bit-for-bit in
    float64 (see tests/test_myadam.py).
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0, bias_correction=True):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay,
                                      bias_correction=bias_correction))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            bc = group["bias_correction"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if not st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["t"] += 1
                t = st["t"]
                if wd:
                    g = g.add(p, alpha=wd)
                m, v = st["m"], st["v"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                if bc:
                    m_hat = m / (1 - b1**t)
                    v_hat = v / (1 - b2**t)
                else:
                    m_hat, v_hat = m, v
                p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)
        return loss
