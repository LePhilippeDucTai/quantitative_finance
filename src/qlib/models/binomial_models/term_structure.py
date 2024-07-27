import numpy as np
from qlib.models.binomial_models.binomial_trees import BinomialTree, FlatForward
from qlib.models.binomial_models.european_options import (
    EuropeanOption,
)


def zero_coupon_bond(maturity: int, term_structure):
    flat_coupon = FlatForward(1.0, maturity)
    return EuropeanOption(maturity, flat_coupon, term_structure)


class ForwardCouponBond:
    def __init__(self, coupon_array: np.ndarray, term_structure: BinomialTree):
        self.coupon_array = np.array(coupon_array)
        self.term_structure = term_structure
        self.q, self.dt = 0.5, 1

    def lattice(self):
        maturities = np.flatnonzero(self.coupon_array)
        coupons = self.coupon_array[maturities]
        zcbs = [zero_coupon_bond(i, self.term_structure) for i in maturities]
        n = max(maturities)
        pads = [n - i for i in maturities]
        return sum(
            np.pad(c * z.lattice(), (0, p)) for c, z, p in zip(coupons, zcbs, pads)
        )

    def npv(self):
        lattice = self.lattice()
        return lattice[0, 0]


class FuturesOption(EuropeanOption):
    def __init__(self, maturity: int, model: BinomialTree):
        super().__init__(maturity, model, None)
        self.term_structure = BinomialTree(0.0, maturity, dt=1, u=0.5, d=0.5)


def main():
    r0 = 0.06
    u, d = 1.25, 0.9
    t_max = 10

    term_structure = BinomialTree(r0, t_max, dt=1, u=u, d=d)
    zcb = zero_coupon_bond(6, term_structure)
    print(zcb.lattice())
    coupon_bond = ForwardCouponBond([0, 0, 0, 0, 0, 0.1, 1.1], term_structure)
    print(coupon_bond.lattice().round(4))

    futures_coupon_bond = FuturesOption(4, model=coupon_bond)
    print(futures_coupon_bond.lattice())


if __name__ == "__main__":
    main()
