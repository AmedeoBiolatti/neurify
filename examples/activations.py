"""Shared activation inspection: python examples/activations.py."""
import neurify as nf


def program(x: nf.Real[-1, 1]):
    with nf.scope("calculation", expose=("a",)):
        a = x + 2
        b = 3 * a
    return b


if __name__ == "__main__":
    net = nf.compile(program)
    assert nf.verify(net).ok
    result = net.inspect(0.5)
    assert result.output == 7.5
    assert result.exposed["calculation"]["a"] == 2.5
    print("Available activations:", dict(net.activation_names))
    print(result)
