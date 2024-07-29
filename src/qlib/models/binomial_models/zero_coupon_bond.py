import numpy as np
from qlib.models.binomial_models.binomial_trees import BinomialTree, FlatForward
from qlib.models.binomial_models.european_options import EuropeanOption


class ZeroCouponBond(EuropeanOption):
    def __init__(self, maturity, term_structure, nominal: float = 1.0):
        model = FlatForward(nominal, maturity)
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


class ForwardZCB:
    def __init__(self, nominal: float, zcb_maturity: int, term_structure: BinomialTree):
        self.nominal = nominal
        self.zcb_maturity = zcb_maturity
        self.term_structure = term_structure

    def _coupon_bonds(self) -> ForwardCouponBond:
        coupon_array = np.zeros(self.zcb_maturity + 1)
        coupon_array[-1] = self.nominal
        return ForwardCouponBond(coupon_array, self.term_structure)

    def forward_price(self, t: int):
        zcb = ZeroCouponBond(t, self.term_structure)
        return self._coupon_bonds().npv() / zcb.npv()


def forward_price_bond(t: int, coupon_array: np.ndarray, term_structure: BinomialTree):
    cb = ForwardCouponBond(coupon_array, term_structure)
    zcb = ZeroCouponBond(t, term_structure)
    return cb.npv() / zcb.npv()
