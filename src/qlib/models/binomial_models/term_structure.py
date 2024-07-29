import numpy as np
from numba import njit
from qlib.models.binomial_models.american_options import (
    AmericanCallOption,
)
from qlib.models.binomial_models.binomial_trees import BinomialTree, FlatForward
from qlib.models.binomial_models.european_options import (
    EuropeanCallOption,
    EuropeanOption,
    rn_expectation,
)
from qlib.utils.logger import logger


def zero_coupon_bond(maturity: int, term_structure):
    flat_coupon = FlatForward(1.0, maturity)
    return EuropeanOption(maturity, flat_coupon, term_structure)


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


class FuturesOption(EuropeanOption):
    def __init__(self, maturity: int, model: BinomialTree):
        super().__init__(maturity, model, None)
        self.term_structure = BinomialTree(0.0, maturity, dt=1, u=0.5, d=0.5)


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


@njit
def arrow_debreu_lattice(term_structure_lattice: np.ndarray, q: float):
    sr = term_structure_lattice
    lattice = np.zeros_like(sr)
    lattice[0, 0] = 1
    n = len(sr)
    for k in range(0, n - 1):
        lattice[k + 1, k + 1] = q * lattice[k, k] / (1 + sr[k, k])
        lattice[0, k + 1] = (1 - q) * lattice[0, k] / (1 + sr[0, k])
        for s in range(1, k + 1):
            left = lattice[s - 1, k] * q / (1 + sr[s - 1, k])
            right = lattice[s, k] * (1 - q) / (1 + sr[s, k])
            lattice[s, k + 1] = left + right
    return lattice


@njit
def compute_agg_induction(
    N, V: np.ndarray, sr_lattice, q: float, dt: float
) -> np.ndarray:
    payoffs = V.copy()
    for j in range(N - 1, 0, -1):
        for i in range(j):
            r = sr_lattice[i, j - 1]
            expected_value = rn_expectation(V[i, j], V[i + 1, j], r, q, dt)
            V[i, j - 1] = payoffs[i, j - 1] + expected_value
    return V


class SwapDerivative(EuropeanOption):
    def __init__(
        self, maturity: int, term_structure: BinomialTree, swap_rate: float, payer: bool
    ):
        """Swap derivative class.
        Payer means paying fixed rate, receiving floating rate. (r - K)
        Payer == False means receive fixed rate, paying floating rate (K - r).
        """
        super().__init__(maturity, term_structure, term_structure)
        self.swap_rate = swap_rate
        self.payer = payer
        self.orient = 1 if payer else -1

    def payoff(self, x: np.ndarray) -> np.ndarray:
        return self.orient * (x - self.swap_rate) / (1 + x)

    def compute_induction(
        self, N: int, V: np.ndarray, short_rate_lattice: np.ndarray, q: float, dt: float
    ) -> np.ndarray:
        return compute_agg_induction(N, V, short_rate_lattice, q, dt)


class Swaption(EuropeanCallOption):
    def __init__(
        self, swap_maturity: int, swaption_maturity, swap_rate: float, term_structure
    ):
        """Swaption.
        Swap_maturity :
        """
        K_strike = 0.0
        self.swaption_maturity = swaption_maturity
        swap_derivative = SwapDerivative(
            maturity=swap_maturity,
            term_structure=term_structure,
            swap_rate=swap_rate,
            payer=True,
        )
        super().__init__(
            maturity=swaption_maturity,
            model=swap_derivative,
            term_structure=term_structure,
            K_strike=K_strike,
        )


class ForwardSwap:
    def __init__(
        self,
        term_structure: BinomialTree,
        swap_rate: float,
        notional: float,
        start: int,
        end: int,
        payer: bool = True,
    ):
        self.term_structure = term_structure
        self.swap_rate = swap_rate
        self.orient = 1 if payer else -1
        self.start = start
        self.end = end
        self.notional = notional

    def payoff(self, r):
        return self.orient * (r - self.swap_rate) / (1 + r)

    def payoff_values(self):
        short_rates_lattice = self.term_structure.lattice()
        return np.triu(self.payoff(short_rates_lattice))

    def npv(self):
        a, b = self.start, self.end
        ts = self.term_structure.lattice()
        q = self.term_structure.q
        ad = arrow_debreu_lattice(ts, q)[:, a:b]
        values = self.payoff_values()[:, a:b]
        print(ad.round(4))
        print(values.round(4))
        return self.notional * (ad * values).sum()


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

    futures_coupon_bond = FuturesOption(4, model=coupon_bond)
    print(futures_coupon_bond.lattice())

    caplet = CapletOption(5, term_structure, term_structure, K_strike=0.02)
    print(caplet.lattice().round(6))

    # Swaps derivatives

    # ts = BinomialTree(0.02, 3, u=1.15, d=0.95)
    # print(ts.lattice())
    term_structure = BinomialTree(r0, 7, dt=1, u=u, d=d)
    ad = arrow_debreu_lattice(term_structure.lattice(), 0.5)
    print(ad.round(4))

    # ZCB(0, 4) from arrow debreu securities
    print(ad[:, 1:4].sum() * 100)

    # Forward-start swap :
    # begins at t = 1, ends at t = 3
    # notional : $1M
    # fixed rate : 7%
    # in arrears, rate seen at t = i - 1 to compute the payment at t = i

    term_structure = BinomialTree(r0, 10, dt=1, u=u, d=d)
    fsw = ForwardSwap(term_structure, 0.07, 1_000_000, 1, 3, payer=False)
    print(fsw.npv())
    # coupon_bond = ForwardCouponBond([0, 0, 0, 100], term_structure)
    # print(coupon_bond.lattice().round(4))

    ts = ShortRateLatticeCustom(
        [
            [0.06, 0.075, 0.0938, 0.1172],
            [0, 0.054, 0.0675, 0.0844],
            [0, 0, 0.0486, 0.0608],
            [0, 0, 0, 0.0437],
        ]
    )
    ad = arrow_debreu_lattice(ts.lattice(), 0.5)
    print(ad.round(4))
    fsw = ForwardSwap(ts, 0.07, 100_000, 1, 3, payer=False)
    print(fsw.npv().round(4))

    swap_derivative = SwapDerivative(
        maturity=5, term_structure=term_structure, swap_rate=0.05, payer=True
    )
    print(swap_derivative.lattice())

    swaption = Swaption(5, 3, 0.05, term_structure)
    print(swaption.lattice())


def assignment():
    ts = BinomialTree(0.05, 10, u=1.1, d=0.9)
    zcb = ZeroCouponBond(10, ts)
    print(np.round(zcb.npv() * 100, 2))

    q2 = forward_price_bond(
        4, coupon_array=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100], term_structure=ts
    )
    print(np.round(q2, 2))

    forward_zcb = ForwardZCB(100, 10, ts)
    print(np.round(forward_zcb.forward_price(4), 2))

    fwd = ForwardCouponBond([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100], ts)
    futures = FuturesOption(4, fwd)
    print(np.round(futures.npv(), 2))

    zcb_call = AmericanCallOption(6, zcb, ts, 0.8)
    print(np.round(zcb_call.npv() * 100, 2))


if __name__ == "__main__":
    main()
    # assignment()
