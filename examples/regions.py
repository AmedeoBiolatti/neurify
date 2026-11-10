"""Run with `python examples/regions.py` after installing Neurify and Torch."""

import neurify as nf


def features(x: nf.Real[-1, 1]):
    shifted = x + 2
    return {"shifted": shifted}


def physics(shifted: nf.Real[1, 3]):
    response = 3 * shifted
    return {"response": response}


def decision(response: nf.Real[3, 9]):
    return 2 * response


def main():
    import torch

    net = nf.pipeline(
        nf.region(features),
        nf.region(physics, trainable=False),
        nf.region(decision),
    )
    assert nf.verify(net).ok
    model = net.to_torch(dtype=torch.float64)
    X = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
    y, before = model.forward_with_intermediates(X, structured=True)
    target = y.detach() + 0.1
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(10):
        optimizer.zero_grad()
        loss = (model(X) - target).square().mean()
        loss.backward()
        optimizer.step()
    _, after = model.forward_with_intermediates(X, structured=True)
    print(net.summary())
    for name in ("features", "physics"):
        key = "shifted" if name == "features" else "response"
        print(f"{name}.{key}: before={before[name][key].tolist()}, after={after[name][key].tolist()}")


if __name__ == "__main__":
    main()
