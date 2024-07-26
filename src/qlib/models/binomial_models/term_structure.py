import numpy as np
from qlib.models.binomial_models.binomial_trees import BinomialTree
from qlib.models.binomial_models.european_options import EuropeanOption


class FlatForward(BinomialTree):
    def __init__(self, s0, T, dt: float = 1.0, u: float = 1, d: float = 1):
        super().__init__(s0, T, dt, u, d)


def to_exponential_rate(r: float) -> float:
    return np.log(1 + r)


def main():
    r0 = 0.06
    u, d = 1.25, 0.9
    t_max = 10
    maturity = 4
    term_structure = BinomialTree(r0, t_max, dt=1, u=u, d=d)
    flat_coupon = FlatForward(1.0, t_max)
    zcb = EuropeanOption(maturity, flat_coupon, term_structure)
    print(term_structure.lattice())
    print(zcb.npv_lattice())


if __name__ == "__main__":
    main()
