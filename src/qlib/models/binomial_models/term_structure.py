import numpy as np
from qlib.models.binomial_models.binomial_trees import BinomialTree, FlatForward
from qlib.models.binomial_models.european_options import (
    EuropeanCallOption,
    EuropeanOption,
)
from qlib.utils.logger import logger


def zero_coupon_bond(maturity: int, term_structure):
    flat_coupon = FlatForward(1.0, maturity)
    return EuropeanOption(maturity, flat_coupon, term_structure)


class ZeroCouponBond(EuropeanOption):
    def __init__(self, maturity, term_structure):
        model = FlatForward(1.0, maturity)
        super().__init__(maturity, model, term_structure)


class ForwardCouponBond:
    def __init__(self, coupon_array: np.ndarray, term_structure: BinomialTree):
        self.coupon_array = np.array(coupon_array)
        self.term_structure = term_structure
        self.q, self.dt = 0.5, 1

    def lattice(self):
        maturities = np.flatnonzero(self.coupon_array)
        coupons = self.coupon_array[maturities]
        zcbs = [ZeroCouponBond(i, self.term_structure) for i in maturities]
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


def forward_price_bond(t: int, coupon_array: np.ndarray, term_structure: BinomialTree):
    cb = ForwardCouponBond(coupon_array, term_structure)
    zcb = ZeroCouponBond(t, term_structure)
    return cb.npv() / zcb.npv()


# print(coupon_bond.lattice().round(4))


class CapletOption(EuropeanCallOption):
    def __init__(
        self,
        maturity: int,
        model: BinomialTree,
        term_structure: BinomialTree,
        K_strike: float,
        in_arears: bool = True,
    ):
        super().__init__(maturity, model, term_structure, K_strike)
        self.in_arears = in_arears

    def payoff(self, x):
        if self.in_arears:
            discount = 1 / (1 + x)
        else:
            discount = 1
        return super().payoff(x) * discount


class ShortRateLatticeCustom:
    def __init__(self, lattice, q=0.5):
        self._lattice = np.array(lattice)
        self.q = q
        self.dt = 1.0

    def lattice(self):
        return self._lattice


def main():
    r0 = 0.06
    u, d = 1.25, 0.9
    t_max = 10

    term_structure = BinomialTree(r0, t_max, dt=1, u=u, d=d)
    zcb = ZeroCouponBond(6, term_structure)
    print(zcb.lattice())

    logger.info("Forward Coupon Bond")
    coupon_array = [0, 0, 0, 0, 0, 10, 110]
    coupon_bond = ForwardCouponBond(coupon_array, term_structure)
    print(coupon_bond.lattice().round(4))
    p0 = forward_price_bond(4, coupon_array, term_structure)
    print(p0)

    futures_coupon_bond = FuturesOption(4, model=coupon_bond)
    print(futures_coupon_bond.lattice())

    caplet = CapletOption(5, term_structure, term_structure, K_strike=0.02)
    print(caplet.lattice().round(6))

    # ts = BinomialTree(0.02, 3, u=1.15, d=0.95)
    # print(ts.lattice())
    ts = ShortRateLatticeCustom(
        [
            [0.02, 0.023, 0.025, 0.026],
            [0, 0.019, 0.021, 0.022],
            [0, 0, 0.018, 0.02],
            [0, 0, 0, 0.015],
        ]
    )

    # Quizz
    zcb = ZeroCouponBond(3, ts)
    print(np.round(zcb.npv(), 4))
    p0 = forward_price_bond(2, [0, 0, 0, 100], ts)
    print(np.round(p0, 4))

    # coupon_bond = ForwardCouponBond([0, 0, 0, 100], term_structure)
    # print(coupon_bond.lattice().round(4))


if __name__ == "__main__":
    main()
