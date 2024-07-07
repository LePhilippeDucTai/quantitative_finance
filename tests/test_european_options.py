import pytest
from qlib.models.black_scholes_model import BlackScholesModel, BlackScholesParameters
from qlib.traits import FlatForward, TermStructure


@pytest.fixture
def bs_model() -> BlackScholesModel:
    rfr = 0.05
    sigma = 0.25
    s0 = 100
    model_params = BlackScholesParameters(s0, sigma)
    term_structure = TermStructure(FlatForward(rfr))
    return BlackScholesModel(model_params, term_structure)


@pytest.fixture
def options_fixtures():
    strike_k = 100
    tmt = 30 / 365
    return tmt, strike_k


def test_deterministic_price(
    bs_model: BlackScholesModel, options_fixtures: tuple[float, float]
):
    tmt, strike_k = options_fixtures
    actual = bs_model.call(tmt, strike_k)
    expected = 3.063
    assert round(actual, 3) == expected

    actual = bs_model.put(tmt, strike_k)
    expected = 2.652
    assert round(actual, 3) == expected
