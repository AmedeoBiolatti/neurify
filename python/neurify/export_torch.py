"""PyTorch export."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class InputNorm(nn.Module):
    """Fixed (non-trainable) input features and normalisation:
    u = (feature(x) - offset) / scale, where a feature is a raw input or a
    one-hot indicator [round(x_i) == v]."""

    def __init__(self, offset, scale, index=None, value=None):
        super().__init__()
        self.register_buffer("offset", torch.as_tensor(offset))
        self.register_buffer("scale", torch.as_tensor(scale))
        n = len(offset)
        index = np.arange(n) if index is None else np.asarray(index)
        value = np.full(n, np.nan) if value is None else np.asarray(value, dtype=np.float64)
        self.register_buffer("index", torch.as_tensor(index, dtype=torch.long))
        self.register_buffer("is_eq", torch.as_tensor(~np.isnan(value)))
        self.register_buffer("value", torch.as_tensor(np.nan_to_num(value), dtype=self.offset.dtype))
        self.identity = bool((index == np.arange(n)).all() and np.isnan(value).all()) if len(index) == n else False

    def forward(self, x):
        f = x if self.identity else x[..., self.index]
        if not self.identity and bool(self.is_eq.any()):
            f = torch.where(self.is_eq, (torch.round(f) == self.value).to(f.dtype), f)
        return (f - self.offset) / self.scale


def _linear(d, dtype):
    w, b = d["weight"], d["bias"]
    lin = nn.Linear(w.shape[1], w.shape[0], dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(torch.as_tensor(w, dtype=dtype))
        lin.bias.copy_(torch.as_tensor(b, dtype=dtype))
    return lin


class ResidualBlock(nn.Module):
    """x + W_out(relu(W_in x) [* W_val x])"""

    def __init__(self, w_in, w_val, w_out):
        super().__init__()
        self.w_in, self.w_val, self.w_out = w_in, w_val, w_out

    def forward(self, x):
        h = torch.relu(self.w_in(x))
        if self.w_val is not None:
            h = h * self.w_val(x)
        return x + self.w_out(h)


class ResidualNet(nn.Module):
    def __init__(self, norm, embed, blocks, readout):
        super().__init__()
        self.norm, self.embed, self.blocks, self.readout = norm, embed, nn.ModuleList(blocks), readout

    def forward(self, x):
        s = self.embed(self.norm(x))
        for b in self.blocks:
            s = b(s)
        return self.readout(s)


def to_torch(net, dtype=None) -> nn.Module:
    dtype = dtype or torch.float32
    w = net.weights()
    if w["body"] == "transformer":
        return TransformerNet(w, dtype)
    norm = InputNorm(
        torch.as_tensor(w["in_offset"], dtype=dtype), torch.as_tensor(w["in_scale"], dtype=dtype),
        w["feat_index"], w["feat_value"],
    )
    if w["body"] == "mlp":
        layers: list[nn.Module] = [norm]
        for h in w["hidden"]:
            layers += [_linear(h, dtype), nn.ReLU()]
        layers.append(_linear(w["out"], dtype))
        return nn.Sequential(*layers)
    blocks = [
        ResidualBlock(
            _linear(b["w_in"], dtype),
            _linear(b["w_val"], dtype) if "w_val" in b else None,
            _linear(b["w_out"], dtype),
        )
        for b in w["blocks"]
    ]
    return ResidualNet(norm, _linear(w["embed"], dtype), blocks, _linear(w["readout"], dtype))


class Attention(nn.Module):
    """Multi-head attention with d_k-dim queries/keys and 1-dim values per head.

    mode: "softmax" (standard), "relu" (unnormalised relu scores) or "hard"
    (argmax; the forward pass is exact, gradients flow only to the values).
    """

    def __init__(self, a, dtype):
        super().__init__()
        self.mode, self.heads, self.d_k = a["mode"], a["heads"], a["d_k"]
        self.q, self.k = _linear(a["w_q"], dtype), _linear(a["w_k"], dtype)
        self.v, self.o = _linear(a["w_v"], dtype), _linear(a["w_o"], dtype)

    def forward(self, x):  # x: (B, T, d)
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.heads, self.d_k)
        k = self.k(x).view(B, T, self.heads, self.d_k)
        v = self.v(x)  # B, T, H
        sc = torch.einsum("bthk,buhk->bhtu", q, k)
        if self.mode == "relu":
            w = torch.relu(sc)
        elif self.mode == "hard":
            w = torch.nn.functional.one_hot(sc.argmax(-1), T).to(x.dtype)
        else:
            w = torch.softmax(sc, -1)
        return self.o(torch.einsum("bhtu,buh->bth", w, v))


class _MLP(nn.Module):
    def __init__(self, b, dtype):
        super().__init__()
        self.w_in = _linear(b["w_in"], dtype)
        self.w_val = _linear(b["w_val"], dtype) if "w_val" in b else None
        self.w_out = _linear(b["w_out"], dtype)

    def forward(self, x):
        h = torch.relu(self.w_in(x))
        if self.w_val is not None:
            h = h * self.w_val(x)
        return self.w_out(h)


def _layernorm(l, d, dtype):
    if l is None:
        return nn.Identity()
    m = nn.LayerNorm(d, eps=l["eps"], dtype=dtype)
    with torch.no_grad():
        m.weight.copy_(torch.as_tensor(l["gain"], dtype=dtype))
        m.bias.copy_(torch.as_tensor(l["bias"], dtype=dtype))
    return m


class TransformerLayer(nn.Module):
    """Pre-LN block: x + attn(ln1(x)); x + mlp(ln2(x)) (each part optional)."""

    def __init__(self, L, d, dtype):
        super().__init__()
        self.ln1 = _layernorm(L.get("ln1"), d, dtype)
        self.attn = Attention(L["attn"], dtype) if "attn" in L else None
        self.ln2 = _layernorm(L.get("ln2"), d, dtype)
        self.mlp = _MLP(L["mlp"], dtype) if "mlp" in L else None

    def forward(self, x):
        if self.attn is not None:
            x = x + self.attn(self.ln1(x))
        if self.mlp is not None:
            x = x + self.mlp(self.ln2(x))
        return x


class TransformerNet(nn.Module):
    """Flat raw inputs -> tokens (learnable per-token embedding, whose bias is
    the positional embedding) -> transformer layers -> readout of token 0."""

    def __init__(self, w, dtype):
        super().__init__()
        off, sc = w["in_offset"], w["in_scale"]
        self.norm = InputNorm(
            torch.as_tensor(off, dtype=dtype), torch.as_tensor(sc, dtype=dtype), w["feat_index"], w["feat_value"]
        )
        E = torch.as_tensor(np.stack([e["weight"] for e in w["tok_embed"]]), dtype=dtype)  # T, d, n_in
        P = torch.as_tensor(np.stack([e["bias"] for e in w["tok_embed"]]), dtype=dtype)  # T, d
        self.tok_embed = nn.Parameter(E)
        self.pos_embed = nn.Parameter(P)
        d = P.shape[1]
        self.layers = nn.ModuleList([TransformerLayer(L, d, dtype) for L in w["layers"]])
        self.ln_f = _layernorm(w["ln_f"], d, dtype)
        self.readout = _linear(w["readout"], dtype)

    def tokens(self, x):
        return torch.einsum("tdi,bi->btd", self.tok_embed, self.norm(x)) + self.pos_embed

    def forward_tokens(self, s):
        for layer in self.layers:
            s = layer(s)
        return self.readout(self.ln_f(s[:, 0]))

    def forward(self, x):
        return self.forward_tokens(self.tokens(x))


class InputFeatures(nn.Module):
    """Raw inputs -> features for the trainable branch: one-hot for integer /
    bool inputs with at most ``onehot_max`` values, [-1, 1]-normalised
    scalars otherwise (all fixed, no parameters)."""

    def __init__(self, specs, onehot_max, dtype):
        super().__init__()
        self.cols = []  # (index, lo, n_values) or (index, offset, scale, None)
        dims = 0
        for i, (_, kind, lo, hi) in enumerate(specs):
            finite = np.isfinite(lo) and np.isfinite(hi)
            if kind != "real" and finite and hi - lo + 1 <= onehot_max:
                self.cols.append((i, float(lo), int(hi - lo) + 1))
                dims += int(hi - lo) + 1
            else:
                off = 0.5 * (lo + hi) if finite else 0.0
                sc = 0.5 * (hi - lo) if finite and hi > lo else 1.0
                self.cols.append((i, off, sc, None))
                dims += 1
        self.dim = dims
        self.dtype = dtype

    def forward(self, x):
        parts = []
        for c in self.cols:
            if len(c) == 3:
                i, lo, n = c
                k = torch.round(x[:, i] - lo).long().clamp(0, n - 1)
                parts.append(nn.functional.one_hot(k, n).to(self.dtype))
            else:
                i, off, sc, _ = c
                parts.append(((x[:, i] - off) / sc).unsqueeze(1).to(self.dtype))
        return torch.cat(parts, 1)


class ExactPlusDelta(nn.Module):
    """``y = exact(x) + delta(features(x))``.

    ``exact`` is the compiled network, frozen (``requires_grad=False``);
    ``delta`` is a plain MLP whose last layer starts at zero, so the model
    starts as the exact program. Trained with standard optimisers on
    ``model.parameters()`` (frozen weights get no gradient), it can only
    *add* to the program; the exact prior cannot be destroyed by large steps.
    """

    def __init__(self, exact, features, hidden, n_out, dtype):
        super().__init__()
        self.exact = exact
        for p in self.exact.parameters():
            p.requires_grad_(False)
        self.features = features
        layers, d = [], features.dim
        for h in hidden:
            layers += [nn.Linear(d, h, dtype=dtype), nn.ReLU()]
            d = h
        last = nn.Linear(d, n_out, dtype=dtype)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.delta = nn.Sequential(*layers, last)

    def forward(self, x):
        with torch.no_grad():
            base = self.exact(x)
        return base + self.delta(self.features(x))


def to_torch_residual(net, hidden=(256, 256), onehot_max=16, dtype=None) -> nn.Module:
    dtype = dtype or torch.float32
    exact = to_torch(net, dtype).eval()
    feats = InputFeatures(net.graph.input_specs(), onehot_max, dtype)
    return ExactPlusDelta(exact, feats, tuple(hidden), net.n_out, dtype)
