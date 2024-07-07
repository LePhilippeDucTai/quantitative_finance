import numpy as np
from qlib.numerical.autodiff import Variable


def test_derivatives():
    val = 3.5
    x = Variable(val)
    assert x.df == 1

    y = x**2
    assert y.df == 3.5 * 2

    z = x.exp().sin()
    assert z.df == np.cos(np.exp(val)) * np.exp(val)


# def test_non_trivial_derivatives():
#     x = Variable(5)
#     y = Variable(3)
#     z = x * y
