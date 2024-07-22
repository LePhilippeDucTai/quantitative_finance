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
