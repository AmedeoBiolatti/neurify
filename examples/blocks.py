"""Configure the same function differently without editing its source.

Run with Torch installed: python examples/blocks.py
"""

import neurify as nf


def program(x: nf.Real[-1, 1]):
    with nf.scope("energy", expose=("energy",)):
        energy = x + 2
    with nf.scope("scoring", expose=("score",)):
        score = 3 * energy + 1
    return score


def main():
    import torch

    torch.set_num_threads(2)
    net = nf.compile(program, blocks={
        "energy": nf.Block(arch="glu"),
        "scoring": nf.Block(residual=()),  # Linear correction; try (32, 32) for an MLP.
    })
    model = net.to_torch(dtype=torch.float64).train_only("scoring")
    print(model.summary())
    function = model.as_function(detach=False)
    print("Before:", model.inspect(x=0.5))
    x = torch.linspace(-1, 1, 129, dtype=torch.float64)
    target = 2 * x + 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    for _ in range(800):
        optimizer.zero_grad(set_to_none=True)
        loss = (function(x=x) - target).square().mean()
        loss.backward()
        optimizer.step()
    mse = (function(x=x) - target).square().mean().item()
    assert mse < 1e-10
    assert all(torch.equal(p, frozen[name]) for name, p in model.named_parameters() if name in frozen)
    print("After: ", model.inspect(x=0.5))
    print(f"Score MSE: {mse:.3g}; frozen parameters unchanged")


if __name__ == "__main__":
    main()
