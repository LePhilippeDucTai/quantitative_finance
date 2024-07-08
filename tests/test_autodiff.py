import numpy as np
from qlib.models.black_scholes_model import BlackScholesModel
from qlib.numerical.autodiff import Variable


def test_derivatives():
    val = 3.5
    x = Variable("x", f=val)
    assert x.grad == 1

    y = x**2
    np.testing.assert_almost_equal(y.grad.data, [3.5 * 2])

    z = x.exp().sin()
    np.testing.assert_almost_equal(z.grad.data, np.cos(np.exp(val)) * np.exp(val))


def test_non_trivial_derivatives_1():
    x = Variable("x", f=3)
    y = Variable("y", f=-1)
    z = Variable("z", f=2)
    f = (((x * y) * (y * z).sin()) ** 2).log()
    np.testing.assert_almost_equal(np.round(f.grad.data, 3), [0.667, -0.169, -0.915])


def test_non_trivial_derivatives_exactly():
    x = Variable("x", f=3)
    y = Variable("y", f=-1)
    z = Variable("z", f=2)
    f = (x * y) * (y * z).sin()
    np.testing.assert_almost_equal(
        f.grad.data,
        [np.sin(2), -3 * (np.sin(2) + 2 * np.cos(2)), 3 * np.cos(2)],
    )


def test_european_call_autodiff(bs_model: BlackScholesModel):
    t = 1.0
    k = 100
    sigma = Variable(id="s", f=0.25)
    s = Variable(id="x", f=100)
    r = Variable(id="r", f=0.05)
    denom = (sigma * t**0.5) ** (-1)
    d1: Variable = denom * ((s / k).log() + (r + 0.5 * sigma**2) * t)
    d2: Variable = d1 - sigma * (t**0.5)
    pricing: Variable = s * d1.ndtr() - k * ((-1) * r * t).exp() * d2.ndtr()

    pricing_det = bs_model.call_det_pricing(t, k)
    np.testing.assert_almost_equal(pricing.grad.data[2], pricing_det.delta)
    np.testing.assert_almost_equal(pricing.grad.data[1], pricing_det.vega)
    np.testing.assert_almost_equal(pricing.f, pricing_det.price)
