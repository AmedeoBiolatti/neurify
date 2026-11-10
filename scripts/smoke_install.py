"""Run with `python -I` to check an installed wheel without a source checkout."""

import argparse
import importlib.metadata
from pathlib import Path

import numpy as np
import neurify as nf


def program(x: nf.Real[-1, 1]):
    with nf.scope("energy", expose=("energy",)):
        energy = x + 2
    with nf.scope("scoring", expose=("score",)):
        score = 3 * energy + 1
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    args = parser.parse_args()
    version = importlib.metadata.version("neurify")
    assert version == nf.__version__ == args.expected_version
    assert "site-packages" in str(Path(nf.__file__).resolve()), nf.__file__
    net = nf.compile(program, blocks={
        "energy": nf.Block(arch="glu", trainable=False),
        "scoring": nf.Block(residual=(8,)),
    })
    x = np.linspace(-1, 1, 17)
    assert np.allclose(net(x), 3 * x + 7)
    assert net.inspect(.5).exposed["energy"]["energy"] == 2.5
    assert nf.verify(net).ok
    # Require the rebuilt bindings, including the native validation changes.
    graph = net.blocks["scoring"].network.graph
    assert hasattr(graph, "output_specs"), "wheel contains an outdated native extension"
    assert graph.output_specs()
    print(f"Installed neurify {version} from {nf.__file__}: compilation, blocks, probes, and native API passed")


if __name__ == "__main__":
    main()
