import numpy as np
import pytest
from qlib.models.binomial_models.american_options import AmericanPutOption
from qlib.models.binomial_models.binomial_trees import BinomialTree, FlatForward
from qlib.models.binomial_models.european_options import (
    EuropeanCallOption,
    EuropeanOption,
)


@pytest.fixture(name="term_structure")
def fixture_simple_term_structure():
    r0 = 0.06
    t_max = 10
    u, d = 1.25, 0.9
    return BinomialTree(r0, t_max, dt=1, u=u, d=d)


@pytest.fixture(name="flat_coupon")
def fixture_simple_flat_coupon():
    t_max = 10
    return FlatForward(1.0, t_max)


@pytest.fixture(name="zcb")
def fixture_simple_zcb(term_structure: BinomialTree, flat_coupon: FlatForward):
    maturity = 4
    return EuropeanOption(maturity, flat_coupon, term_structure)


def test_zero_coupon_bond(zcb: EuropeanOption):
    np.testing.assert_almost_equal(zcb.npv(), 0.7721774)


def test_zero_coupon_bond_call(zcb: EuropeanOption, term_structure: BinomialTree):
    zcb_call = EuropeanCallOption(2, zcb, term_structure, 0.84)
    np.testing.assert_almost_equal(zcb_call.npv(), 0.02969474)


def test_zero_coupon_bond_american_put(
    zcb: EuropeanOption, term_structure: BinomialTree
):
    zcb_call = AmericanPutOption(2, zcb, term_structure, 0.88)
    np.testing.assert_almost_equal(zcb_call.npv(), 0.1078226)
