from enum import Enum, auto


class ComputationKind(Enum):
    DET = auto()
    EULER = auto()
    EULER_JIT = auto()
    EXACT = auto()
    MILTSTEIN = auto()
    TERMINAL = auto()


def first_order_derivative(h, f1, fm1):
    return (f1 - fm1) / (2 * h)


def second_order_derivative(h: float, f2: float, f1, f0, fm1, fm2):
    return (-f2 + 16 * f1 - 30 * f0 + 16 * fm1 - fm2) / (12 * h**2)
