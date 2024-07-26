import numpy as np
from qlib.models.binomial_models.american_options import (
    AmericanCallOption,
    AmericanPutOption,
)
from qlib.models.binomial_models.binomial_trees import CRRModel, FlatForward
from qlib.models.binomial_models.european_options import (
    EuropeanCallOption,
    EuropeanPutOption,
)


def test_binomial_models():
    r, sigma, T, dt = 0.3, 0.5, 3, 0.01
    s0 = 100
    K = 110
    term_structure = FlatForward(r, T, dt)
    model = CRRModel(s0, sigma, term_structure, T, dt)
    call = EuropeanCallOption(T, model, term_structure, K)
    put = EuropeanPutOption(T, model, term_structure, K)
    np.testing.assert_almost_equal(put.npv(), 5.218352546678223)
    np.testing.assert_almost_equal(call.npv(), 60.36132053411522)


def test_binomial_model_american():
    r, sigma, T, dt = 0.3, 0.5, 3, 0.01
    s0 = 100
    K = 110
    term_structure = FlatForward(r, T, dt)
    model = CRRModel(s0, sigma, term_structure, T, dt)
    call = AmericanCallOption(T, model, term_structure, K)
    put = AmericanPutOption(T, model, term_structure, K)

    np.testing.assert_almost_equal(put.npv(), 17.22294182198558)
    np.testing.assert_almost_equal(call.npv(), 60.36132053411522)
    # Same price as the European one
