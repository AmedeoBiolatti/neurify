"""Run after installing Neurify: python examples/scopes.py."""
import neurify as nf


def program(x: nf.Real[-1, 1]):
    with nf.scope("physics", trainable=False, expose=("energy",)):
        energy = x + 2
    with nf.scope("decision"):
        score = 3 * energy
    return score + x


if __name__ == "__main__":
    net = nf.compile(program)
    assert nf.verify(net).ok
    output, values = net.forward_with_intermediates([[0.5], [0.0]])
    print(net.summary())
    print("Output:", output)
    print("Energy:", values["physics"]["energy"])
