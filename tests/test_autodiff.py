import numpy as np
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
    np.testing.assert_equal(
        np.round(f.grad.data, 3),
        [np.sin(2), -3 * (np.sin(2) + 2 * np.cos(2)), 3 * np.cos(2)],
    )
