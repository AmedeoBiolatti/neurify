"""JAX export: a pure function and a params pytree."""

from __future__ import annotations

import jax.numpy as jnp


def to_jax(net, dtype=None):
    """Returns (apply, params). ``apply(params, x)`` maps (batch, n_in) or
    (n_in,) raw inputs to outputs; input normalisation is a fixed constant,
    not a parameter."""
    dtype = dtype or jnp.float32
    w = net.weights()
    import numpy as _np

    idx = _np.asarray(w["feat_index"])
    val = _np.asarray(w["feat_value"])
    eq = ~_np.isnan(val)
    identity = len(idx) == w["in_offset"].shape[0] and not eq.any() and (idx == _np.arange(len(idx))).all()
    off0 = jnp.asarray(w["in_offset"], dtype)
    sc0 = jnp.asarray(w["in_scale"], dtype)
    eqj, valj = jnp.asarray(eq), jnp.asarray(_np.nan_to_num(val), dtype)

    def norm(x):
        """fixed input features, normalised: (feature(x) - offset) / scale"""
        f = x if identity else x[..., idx]
        if not identity and eq.any():
            f = jnp.where(eqj, (jnp.round(f) == valj).astype(f.dtype), f)
        return (f - off0) / sc0

    def dense(d):
        return {"w": jnp.asarray(d["weight"], dtype), "b": jnp.asarray(d["bias"], dtype)}

    if w["body"] == "mlp":
        params = {"hidden": [dense(h) for h in w["hidden"]], "out": dense(w["out"])}

        def apply(p, x):
            h = norm(x)
            for d in p["hidden"]:
                h = jnp.maximum(h @ d["w"].T + d["b"], 0)
            return h @ p["out"]["w"].T + p["out"]["b"]

        return apply, params

    if w["body"] == "transformer":
        return _transformer(w, dtype, norm, dense)

    params = {
        "embed": dense(w["embed"]),
        "blocks": [{k: dense(v) for k, v in b.items()} for b in w["blocks"]],
        "readout": dense(w["readout"]),
    }

    def apply(p, x):
        s = (norm(x)) @ p["embed"]["w"].T + p["embed"]["b"]
        for b in p["blocks"]:
            h = jnp.maximum(s @ b["w_in"]["w"].T + b["w_in"]["b"], 0)
            if "w_val" in b:
                h = h * (s @ b["w_val"]["w"].T + b["w_val"]["b"])
            s = s + h @ b["w_out"]["w"].T + b["w_out"]["b"]
        return s @ p["readout"]["w"].T + p["readout"]["b"]

    return apply, params


def _transformer(w, dtype, norm, dense):
    import jax
    import numpy as np

    def ln_p(l):
        return None if l is None else {"g": jnp.asarray(l["gain"], dtype), "b": jnp.asarray(l["bias"], dtype)}

    eps = {}
    layers, meta = [], []
    for L in w["layers"]:
        p, m = {}, {}
        for k in ("ln1", "ln2"):
            if k in L:
                p[k] = ln_p(L[k])
                eps[k] = L[k]["eps"]
        if "attn" in L:
            a = L["attn"]
            p["attn"] = {k: dense(a[k]) for k in ("w_q", "w_k", "w_v", "w_o")}
            m["attn"] = (a["mode"], a["heads"], a["d_k"])
        if "mlp" in L:
            p["mlp"] = {k: dense(v) for k, v in L["mlp"].items()}
        layers.append(p)
        meta.append(m)
    params = {
        "tok_embed": jnp.asarray(np.stack([e["weight"] for e in w["tok_embed"]]), dtype),
        "pos_embed": jnp.asarray(np.stack([e["bias"] for e in w["tok_embed"]]), dtype),
        "layers": layers,
        "ln_f": ln_p(w["ln_f"]),
        "readout": dense(w["readout"]),
    }
    ln_eps = next(iter(eps.values()), 1e-5)

    def ln(p, x):
        if p is None:
            return x
        m = x.mean(-1, keepdims=True)
        v = ((x - m) ** 2).mean(-1, keepdims=True)
        return p["g"] * (x - m) / jnp.sqrt(v + ln_eps) + p["b"]

    def attn(p, m, x):
        mode, H, K = m
        B, T, _ = x.shape
        q = (x @ p["w_q"]["w"].T + p["w_q"]["b"]).reshape(B, T, H, K)
        k = (x @ p["w_k"]["w"].T + p["w_k"]["b"]).reshape(B, T, H, K)
        v = x @ p["w_v"]["w"].T + p["w_v"]["b"]
        s = jnp.einsum("bthk,buhk->bhtu", q, k)
        if mode == "relu":
            wt = jnp.maximum(s, 0)
        elif mode == "hard":
            wt = jax.nn.one_hot(jnp.argmax(s, -1), T, dtype=x.dtype)
        else:
            wt = jax.nn.softmax(s, -1)
        o = jnp.einsum("bhtu,buh->bth", wt, v)
        return o @ p["w_o"]["w"].T + p["w_o"]["b"]

    def mlp(p, x):
        h = jnp.maximum(x @ p["w_in"]["w"].T + p["w_in"]["b"], 0)
        if "w_val" in p:
            h = h * (x @ p["w_val"]["w"].T + p["w_val"]["b"])
        return h @ p["w_out"]["w"].T + p["w_out"]["b"]

    def apply(p, x):
        single = x.ndim == 1
        x = jnp.atleast_2d(x)
        u = norm(x)
        s = jnp.einsum("tdi,bi->btd", p["tok_embed"], u) + p["pos_embed"]
        for L, m in zip(p["layers"], meta):
            if "attn" in L:
                s = s + attn(L["attn"], m["attn"], ln(L.get("ln1"), s))
            if "mlp" in L:
                s = s + mlp(L["mlp"], ln(L.get("ln2"), s))
        y = ln(p["ln_f"], s[:, 0]) @ p["readout"]["w"].T + p["readout"]["b"]
        return y[0] if single else y

    return apply, params


def to_jax_residual(net, hidden=(256, 256), onehot_max=16, seed=0, dtype=None):
    """(apply, params) for ``y = exact(x) + delta(features(x))``: ``params``
    holds only the trainable branch (last layer zero), the exact network is a
    constant behind ``stop_gradient``. See ``export_torch.ExactPlusDelta``."""
    import jax
    import numpy as np

    dtype = dtype or jnp.float32
    exact_apply, exact_params = to_jax(net, dtype)
    specs = net.graph.input_specs()
    cols, dim = [], 0
    for i, (_, kind, lo, hi) in enumerate(specs):
        finite = np.isfinite(lo) and np.isfinite(hi)
        if kind != "real" and finite and hi - lo + 1 <= onehot_max:
            cols.append((i, float(lo), int(hi - lo) + 1))
            dim += int(hi - lo) + 1
        else:
            off = 0.5 * (lo + hi) if finite else 0.0
            sc = 0.5 * (hi - lo) if finite and hi > lo else 1.0
            cols.append((i, off, sc, None))
            dim += 1

    def features(x):
        parts = []
        for c in cols:
            if len(c) == 3:
                i, lo, n = c
                k = jnp.clip(jnp.round(x[:, i] - lo).astype(jnp.int32), 0, n - 1)
                parts.append(jax.nn.one_hot(k, n, dtype=dtype))
            else:
                i, off, sc, _ = c
                parts.append(((x[:, i] - off) / sc)[:, None].astype(dtype))
        return jnp.concatenate(parts, 1)

    rng = np.random.default_rng(seed)
    params, d = [], dim
    for h in hidden:
        bound = 1.0 / np.sqrt(d)
        params.append({"w": jnp.asarray(rng.uniform(-bound, bound, (h, d)), dtype), "b": jnp.zeros(h, dtype)})
        d = h
    params.append({"w": jnp.zeros((net.n_out, d), dtype), "b": jnp.zeros(net.n_out, dtype)})

    def apply(p, x):
        x = jnp.atleast_2d(x)
        base = jax.lax.stop_gradient(exact_apply(exact_params, x))
        h = features(x)
        for layer in p[:-1]:
            h = jnp.maximum(h @ layer["w"].T + layer["b"], 0)
        return base + h @ p[-1]["w"].T + p[-1]["b"]

    return apply, params
