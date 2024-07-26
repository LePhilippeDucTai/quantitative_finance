from qlib.models.binomial_models.american_options import AmericanPutOption
from qlib.models.binomial_models.binomial_trees import BinomialTree, FlatForward
from qlib.models.binomial_models.european_options import (
    EuropeanCallOption,
    EuropeanOption,
)


def main():
    r0 = 0.06
    u, d = 1.25, 0.9
    t_max = 10
    maturity = 4

    term_structure = BinomialTree(r0, t_max, dt=1, u=u, d=d)
    flat_coupon = FlatForward(1.0, t_max)
    zcb = EuropeanOption(maturity, flat_coupon, term_structure)
    zcb_call = EuropeanCallOption(2, zcb, term_structure, 0.84)
    print(zcb.lattice())
    print(zcb_call.lattice())
    zcb_american_put = AmericanPutOption(3, zcb, term_structure, 0.88)
    print(zcb_american_put.lattice())


if __name__ == "__main__":
    main()
