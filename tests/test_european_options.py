from qlib.models.black_scholes_model import BlackScholesModel


def bs_parameters_fixtures():
    rfr = 0.05
    sigma = 0.25
    s0 = 100
    strike_k = 100
    tmt = 30 / 365
    return rfr, sigma, s0, strike_k, tmt


def test_deterministic_price():
    rfr, sigma, s0, strike_k, tmt = bs_parameters_fixtures()
    bs_model_det = BlackScholesModel(s0, tmt, rfr, sigma)
    actual = bs_model_det.call(strike_k)
    expected = 3.063
    assert round(actual, 3) == expected

    actual = bs_model_det.put(strike_k)
    expected = 2.652
    assert round(actual, 3) == expected
