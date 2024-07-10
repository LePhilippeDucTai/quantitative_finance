from typing import Self

import numpy as np
import xarray as xa
from scipy.special import ndtr
from scipy.stats import norm


def make_gradient(grad: float, id: str | list):
    ids = list(id)
    n = len(ids)
    return xa.DataArray([grad] * n, coords={"a": list(id)})


def align_arrays(left: xa.DataArray, right: xa.DataArray) -> xa.DataArray:
    return xa.align(left, right, join="outer", fill_value=0.0)


def extract_id(grad: xa.DataArray):
    return grad.coords["a"].values.tolist()


def make_scalar(id, f: float):
    return Variable(id, f, 0.0)


class Variable:
    def __init__(self, id: str, f: float = 1.0, grad: float = 1.0):
        self.id = [id] if isinstance(id, str) else id
        self.f = f
        self.grad = make_gradient(grad, id) if isinstance(grad, float) else grad

    def __repr__(self):
        return f"Var[{self.id}](f={self.f}, grad={self.grad.data})"

    def __add__(self, other: Self):
        if isinstance(other, float) or isinstance(other, int):
            other = make_scalar(self.id, f=other)
        f = self.f + other.f
        df, dg = align_arrays(self.grad, other.grad)
        grad = df + dg
        id = extract_id(grad)
        return Variable(id, f, grad)

    def __radd__(self, other: Self | float):
        return other + self

    def __mul__(self, other: Self):
        if isinstance(other, float) or isinstance(other, int):
            other = make_scalar(self.id, f=other)
        f = self.f * other.f
        df, dg = align_arrays(self.grad, other.grad)
        grad = df * other.f + dg * self.f
        id = extract_id(grad)
        return Variable(id, f, grad)

    def __rmul__(self, other: Self | float):
        return self * other

    def __truediv__(self, other: Self | float):
        if isinstance(other, float) or isinstance(other, int):
            other = make_scalar(self.id, f=other)
        f = self.f / other.f
        df, dg = align_arrays(self.grad, other.grad)
        grad = (df * other.f - dg * self.f) / (other.f**2)
        id = extract_id(grad)
        return Variable(id, f, grad)

    def __rtruediv__(self, other: Self | float):
        return other.__truediv__(self)

    def __sub__(self, other: Self | float | int):
        return self + (-1) * other

    def __rsub__(self, other: Self | float | int):
        return other - self

    def __neg__(self):
        return Variable(self.id, -self.f, -self.grad)

    def exp(self) -> Self:
        f = np.exp(self.f)
        grad = np.exp(self.f) * self.grad
        return Variable(self.id, f, grad)

    def sin(self) -> Self:
        f = np.sin(self.f)
        grad = np.cos(self.f) * self.grad
        return Variable(self.id, f, grad)

    def __pow__(self, n: float) -> Self:
        f = self.f**n
        grad = n * self.f ** (n - 1) * self.grad
        return Variable(self.id, f, grad)

    def log(self) -> Self:
        f = np.log(self.f)
        grad = self.grad / self.f
        return Variable(self.id, f, grad)

    def sqrt(self):
        f = np.sqrt(self.f)
        grad = 0.5 * self.grad / f
        return Variable(self.id, f, grad)

    def ndtr(self) -> Self:
        f = ndtr(self.f)
        grad = norm.pdf(self.f) * self.grad
        return Variable(self.id, f, grad)


def main():
    x = Variable("x", f=3)
    y = Variable("y", f=-1)
    z = Variable("z", f=2)
    a = Variable("a", f=0.5)
    f = (
        ((x * y) * (y * z).sin().exp().sqrt()).exp()
        - x / z * y.exp()
        - (-x + z / a).sin()
        + 32 * (a + x**2).sin()
    )
    print(f)


if __name__ == "__main__":
    main()
