"""neurify: compile plain Python functions into exactly-equivalent,
trainable neural networks.

    import neurify as nf

    def f(x: nf.Int[0, 15], y: nf.Int[0, 15]):
        if x > y:
            return (x * y) % 7
        return max(x - y, 3)

    net = nf.compile(f, arch="mlp", method="depth")
    net(9, 4)                 # == f(9, 4), computed by the network
    nf.verify(net)            # exhaustive check on the declared domain
    model = net.to_torch()    # exact initialisation, ready to fine-tune
"""

from ._core import CompileError, TraceError
from .compiler import ARCHS, ATTENTIONS, METHODS, OUTPUTS, compile
from .blocks import Block
from .domains import Bool, Domain, Int, Real, Seq, Static, Vec
from .merge import COMBINES, merge
from .network import Network
from .ops import relu, where
from .regions import Pipeline, Region, pipeline, region
from .scopes import Scope, ScopedNetwork, scope
from .interface import Evaluation
from .trainability import TrainabilityReport, trainability_report
from .verify import VerifyResult, verify

__all__ = [
    "compile", "merge", "COMBINES", "verify", "Network", "VerifyResult", "trainability_report", "TrainabilityReport",
    "Real", "Int", "Bool", "Vec", "Seq", "Static", "Domain",
    "where", "relu", "CompileError", "TraceError", "ARCHS", "ATTENTIONS", "METHODS", "OUTPUTS",
    "region", "pipeline", "Region", "Pipeline",
    "scope", "Scope", "ScopedNetwork",
    "Evaluation", "Block",
]
__version__ = "0.1.0"
