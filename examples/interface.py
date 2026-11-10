"""Function-style prediction and structured inspection without optional backends."""
import neurify as nf


def program(x: nf.Real[-1, 1], *, shift: nf.Real[0, 2] = 1):
    with nf.scope("physics", expose=("energy",)):
        energy = x + shift
    return {"score": 2 * energy, "energy": energy}


if __name__ == "__main__":
    net = nf.compile(program)
    print("Prediction:", net(0.5))
    result = net.inspect(0.5)
    print("Output:", result.output)
    print("Exposed:", result.exposed)
    assert result.output == {"score": 3.0, "energy": 1.5}
    assert result.exposed == {"physics": {"energy": 1.5}}
