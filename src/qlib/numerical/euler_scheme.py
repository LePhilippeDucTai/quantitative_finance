from enum import Enum, auto


class ComputationKind(Enum):
    DET = auto()
    EULER = auto()
    EULER_JIT = auto()
    EXACT = auto()
    MILTSTEIN = auto()
    TERMINAL = auto()
