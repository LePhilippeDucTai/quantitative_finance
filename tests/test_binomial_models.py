from qlib.models.binomial_models.american_options import (
    AmericanCallOption,
    AmericanPutOption,
)
from qlib.models.binomial_models.binomial_trees import CRRModel
from qlib.models.binomial_models.european_options import (
    EuropeanCallOption,
    EuropeanPutOption,
)


def test_binomial_models():
    r, sigma, T, dt = 0.3, 0.5, 3, 0.01
    s0 = 100
    K = 110
    model = CRRModel(s0, r, sigma, T, dt)
    call = EuropeanCallOption(model, K)
    put = EuropeanPutOption(model, K)

    assert put.npv() == 5.218352546678223
    assert call.npv() == 60.36132053411522


def test_binomial_model_american():
    r, sigma, T, dt = 0.3, 0.5, 3, 0.01
    s0 = 100
    K = 110
    model = CRRModel(s0, r, sigma, T, dt)
    call = AmericanCallOption(model, K)
    put = AmericanPutOption(model, K)

    assert put.npv() == 17.22294182198558
    assert call.npv() == 60.36132053411522  # Same price as the European one
