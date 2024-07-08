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
