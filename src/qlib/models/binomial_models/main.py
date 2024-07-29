import numpy as np
from qlib.models.binomial_models.american_options import AmericanCallOption
from qlib.models.binomial_models.binomial_trees import BinomialTree
from qlib.models.binomial_models.rates_options import (
    CapletOption,
    ForwardSwap,
    FuturesOption,
    SwapDerivative,
    Swaption,
)
from qlib.models.binomial_models.term_structure import (
    ShortRateLatticeCustom,
    arrow_debreu_lattice,
)
from qlib.models.binomial_models.zero_coupon_bond import (
    ForwardCouponBond,
    ForwardZCB,
    ZeroCouponBond,
    forward_price_bond,
)
from qlib.utils.logger import logger


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
