"""An exact registered lowering with four trainable, zero-output residual blocks.

Run with the Torch extra installed: python examples/classical_transformer.py
This is a deliberately small backend example, not a generic shallow compiler.
"""
import numpy as np
import neurify as nf


def score(enabled: nf.Bool):
    return 30 if enabled else 10


@nf.transformer_lowering(score, exact=True)
def lower_score(*, target, reference, inputs, static):
    import torch
    from torch import nn

    specs = reference.graph.input_specs()
    if static or len(specs) != 1 or tuple(specs[0][1:]) != ("bool", 0, 1):
        raise nf.CompileError("this example lowering supports one Boolean input")
    shape = target.resolve(required_blocks=4, min_d_model=4, ffn_widths=[1] * 4)

    class ScoreModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    shape["d_model"], target.heads, shape["ffn_widths"][i],
                    dropout=0., activation="relu", batch_first=True,
                    norm_first=True, dtype=torch.float64,
                ) for i in range(4)
            ])
            self.readout = nn.Linear(shape["d_model"], 1, dtype=torch.float64)
            with torch.no_grad():
                for layer in self.layers:
                    # Keep random incoming features; each residual write is zero.
                    layer.self_attn.out_proj.weight.zero_()
                    layer.self_attn.out_proj.bias.zero_()
                    layer.linear2.weight.zero_()
                    layer.linear2.bias.zero_()
                self.readout.weight.zero_()
                self.readout.weight[0, 0] = 10
                self.readout.bias.fill_(20)

        def forward(self, x):
            signed = 2 * x[:, :1] - 1
            state = torch.cat([signed, -signed] * (shape["d_model"] // 2), -1).unsqueeze(1)
            for layer in self.layers:
                state = layer(state)
            return self.readout(state[:, 0])

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        return ScoreModel()


def main():
    import torch
    torch.set_num_threads(2)
    net = nf.compile(score, arch=nf.Transformer(), strict=True)
    np.testing.assert_array_equal(net(enabled=np.array([False, True])), [10, 30])
    assert nf.verify(net).ok
    model = net.to_torch(dtype=torch.float64)
    x = torch.tensor([[0.], [1.]], dtype=torch.float64)
    (model(x) - torch.tensor([[12.], [35.]], dtype=torch.float64)).square().mean().backward()
    assert any(bool(layer.linear2.weight.grad.abs().sum()) for layer in model.model.layers)
    print(net.summary())
    print("Exact initial scores:", net(enabled=np.array([False, True])))
    print("The zero-output FFNs receive gradients on the first backward pass.")


if __name__ == "__main__":
    main()
