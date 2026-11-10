"""How trainable is a compiled network with *standard* methods?

``trainability_report(net)`` measures, on samples of the declared domain:

* **gradient reach**: which parameters get a non-zero gradient from a generic
  loss (a random linear functional of the outputs). Exact discrete gadgets are
  flat at every admissible value, so only layers after the last such gadget
  are reached.
* **Adam tolerance**: plain Adam on a pure-noise objective (a fresh random
  functional of the outputs at every step, like the early steps of RL or of
  fine-tuning on a new task). For each learning rate, after ``steps`` steps,
  the fraction of samples on which the network still computes the program.
  ``adam_tolerance`` is the largest learning rate that keeps >= ``keep``.
* **bottleneck**: per layer, the largest random ±δ perturbation of all its
  pre-activations that keeps the program exact (``z_tol``), and the learning
  rate this predicts, ``z_tol / (steps * (|x|_1 + 1))``: one Adam step moves
  a pre-activation by about ``lr * (|x|_1 + 1)`` (every weight with a gradient
  moves by ~lr, zero weights included). The smallest one is the bottleneck.
* **scales**: per layer, the fraction of zero weights, the He ratio
  ``std(W) * sqrt(fan_in / 2)`` (1 for He init), max |w|, activation RMS.

A network that is "natural" (easy to train) reaches all layers with
gradient, tolerates the learning rates people actually use (1e-4 .. 1e-3)
and has He ratios and activations of order 1.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

LRS = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)


@dataclass
class TrainabilityReport:
    grad_reach: float  # fraction of layers (weight matrices) with some non-zero gradient
    first_reached_layer: int  # index of the first weight matrix with a non-zero gradient
    grad_coverage: float  # fraction of all parameters with non-zero gradient
    adam_tolerance: float  # largest lr keeping >= keep exact after `steps` noise steps (0 if none)
    adam_curve: dict  # lr -> fraction of samples still exact
    layers: list = field(default_factory=list)  # per weight matrix stats
    steps: int = 10
    keep: float = 0.99
    bottleneck: int = -1
    lr_pred: float = 0.0

    def __str__(self) -> str:
        s = [
            f"gradient reaches {self.grad_reach:.0%} of the layers (first: {self.first_reached_layer}), "
            f"{self.grad_coverage:.2%} of the parameters",
            f"Adam tolerance ({self.steps} noise steps, keep {self.keep:.0%}): "
            + (f"lr <= {self.adam_tolerance:g}" if self.adam_tolerance else "none"),
            "  " + "  ".join(f"{lr:g}:{v:.2f}" for lr, v in self.adam_curve.items()),
            f"bottleneck: layer {self.bottleneck} (predicted lr <= {self.lr_pred:.2g})",
            f"{'layer':>5} {'shape':>12} {'zeros':>6} {'He ratio':>8} {'max|w|':>8} {'act rms':>8} {'|x|_1':>8} "
            f"{'z_tol':>8} {'lr_pred':>8} {'grad':>5}",
        ]
        for i, l in enumerate(self.layers):
            s.append(
                f"{i:5d} {str(l['shape']):>12} {l['zeros']:6.1%} {l['he_ratio']:8.3g} {l['max_w']:8.3g} "
                f"{l['act_rms']:8.3g} {l['x_l1']:8.3g} {l['z_tol']:8.2g} {l['lr_pred']:8.2g} "
                f"{'yes' if l['grad'] else '-':>5}"
            )
        return "\n".join(s)


def _match(y, y_ref, int_like, match):
    if match == "argmax":
        best = y_ref == y_ref.max(1, keepdims=True)
        pick = y.argmax(1)
        return best[np.arange(len(y)), pick]
    tol = np.where(int_like, 0.25, 1e-3 * (1 + np.abs(y_ref)))
    return np.all(np.abs(y - y_ref) <= tol, axis=1)


def trainability_report(
    net,
    X=None,
    *,
    n: int = 512,
    steps: int = 10,
    keep: float = 0.99,
    lrs=LRS,
    match: str = "value",
    seed: int = 0,
    model=None,
) -> TrainabilityReport:
    """Measure gradient reach, Adam tolerance and scales (see module doc).

    ``X``: raw input batch (default: ``n`` samples of the declared domain).
    ``match``: ``"value"`` (outputs within 0.25 for integer outputs, else
    relative 1e-3) or ``"argmax"`` (the network's argmax is a maximum of the
    program's outputs). ``model``: an exported torch module to measure instead
    of ``net.to_torch()`` (e.g. after ``prepare_for_training``).
    """
    import torch

    if X is None:
        from .verify import _sample

        X, _ = _sample(net.graph.input_specs(), n, np.random.default_rng(seed), n)
    X = np.asarray(X, dtype=np.float64)
    y_ref, _ = net.reference(X)
    kinds = [s[1] for s in _output_kinds(net)]
    int_like = np.array([k != "real" for k in kinds]) if len(kinds) == y_ref.shape[1] else np.zeros(y_ref.shape[1], bool)
    m0 = (model if model is not None else net.to_torch()).float().eval()
    xt = torch.as_tensor(X, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    mats = [(name, p) for name, p in m0.named_parameters() if p.dim() == 2]

    # gradient reach: a random linear functional of the outputs
    m0.zero_grad()
    y = m0(xt)
    (torch.randn(y.shape, generator=g) * y).sum().backward()
    reached = [bool((p.grad is not None) and (p.grad != 0).any()) for _, p in mats]
    params = [p for p in m0.parameters()]
    total = sum(p.numel() for p in params)
    covered = sum(int((p.grad != 0).sum()) for p in params if p.grad is not None)

    # activations: forward hooks on every Linear
    acts = {}
    hooks = []
    for name, mod in m0.named_modules():
        if isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(lambda m, i, o, k=name: acts.__setitem__(k, o.detach())))
    with torch.no_grad():
        m0(xt)
    for h in hooks:
        h.remove()
    layers = []
    for (name, p), r in zip(mats, reached):
        w = p.detach()
        fan_in = w.shape[1]
        a = acts.get(name.rsplit(".", 1)[0])
        layers.append({
            "name": name,
            "shape": tuple(w.shape),
            "zeros": float((w == 0).float().mean()),
            "he_ratio": float(w.std() * np.sqrt(fan_in / 2)),
            "max_w": float(w.abs().max()),
            "act_rms": float(torch.relu(a).pow(2).mean().sqrt()) if a is not None else float("nan"),
            "grad": r,
        })

    # per-layer pre-activation noise tolerance (bisection on a log grid)
    lin_names = [name.rsplit(".", 1)[0] for name, _ in mats]
    modules = dict(m0.named_modules())
    for li, (lname, layer) in enumerate(zip(lin_names, layers)):
        mod = modules.get(lname)
        if mod is None:
            layer.update(z_tol=float("nan"), lr_pred=float("nan"), x_l1=float("nan"))
            continue
        xin = {}
        h = mod.register_forward_hook(lambda m, i, o: xin.__setitem__("x", i[0].detach()))
        with torch.no_grad():
            m0(xt)
        h.remove()
        x_l1 = float(xin["x"].abs().sum(-1).mean()) if "x" in xin else float("nan")
        sign = torch.sign(torch.randn(mod.out_features, generator=g))

        def exact_with(delta):
            hk = mod.register_forward_hook(lambda m, i, o: o + delta * sign)
            with torch.no_grad():
                y = m0(xt).double().numpy()
            hk.remove()
            return np.mean(_match(y, y_ref, int_like, match)) >= keep

        lo, hi = 0.0, None
        for k in range(-9, 3):
            if exact_with(10.0**k):
                lo = 10.0**k
            else:
                hi = 10.0**k
                break
        if hi is not None and lo > 0:
            for _ in range(6):
                mid = np.sqrt(lo * hi)
                lo, hi = (mid, hi) if exact_with(mid) else (lo, mid)
        layer.update(z_tol=lo, x_l1=x_l1, lr_pred=lo / (steps * (x_l1 + 1)))
    preds = [l["lr_pred"] for l in layers]
    bottleneck = int(np.nanargmin(preds)) if preds and not np.all(np.isnan(preds)) else -1

    # Adam on pure noise
    curve = {}
    for lr in lrs:
        m = copy.deepcopy(m0).train()
        opt = torch.optim.Adam(m.parameters(), lr=lr)
        gl = torch.Generator().manual_seed(seed + 1)
        for _ in range(steps):
            opt.zero_grad()
            y = m(xt)
            (torch.randn(y.shape, generator=gl) * y).mean().backward()
            opt.step()
        with torch.no_grad():
            y = m(xt).double().numpy()
        curve[lr] = float(np.mean(_match(y, y_ref, int_like, match)))
    ok = [lr for lr in lrs if curve[lr] >= keep]
    # tolerance = largest lr such that it and every smaller lr keep enough
    tol = 0.0
    for lr in lrs:
        if curve[lr] >= keep:
            tol = lr
        else:
            break
    del ok
    first = next((i for i, r in enumerate(reached) if r), len(reached))
    return TrainabilityReport(
        grad_reach=float(np.mean(reached)) if reached else 0.0,
        first_reached_layer=first,
        grad_coverage=covered / max(total, 1),
        adam_tolerance=tol,
        adam_curve=curve,
        layers=layers,
        steps=steps,
        keep=keep,
        bottleneck=bottleneck,
        lr_pred=float(preds[bottleneck]) if bottleneck >= 0 else 0.0,
    )


def _output_kinds(net):
    """[(name, kind)] of the flat outputs, from the output spec."""
    out = []

    def walk(spec):
        tag = spec[0]
        if tag == "leaf":
            out.append(("", spec[1]))
        elif tag in ("list", "tuple"):
            for s in spec[1]:
                walk(s)
        elif tag == "dict":
            for _, s in spec[1]:
                walk(s)

    walk(net.out_spec)
    return out
