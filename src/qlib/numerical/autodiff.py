from typing import Self

import numpy as np
import xarray as xa
from scipy.special import ndtr
from scipy.stats import norm


def make_gradient(grad: float, id: str | list):
    return xa.DataArray([grad], coords={"a": list(id)})


class Variable:
    def __init__(self, id: str, f: float = 1.0, grad: float = 1.0):
        self.id = [id] if isinstance(id, str) else id
        self.f = f
        self.grad = make_gradient(grad, id) if isinstance(grad, float) else grad

    def __repr__(self):
        return f"Var[{self.id}](f={self.f}, grad={self.grad.data})"

    def __add__(self, other: Self):
        if isinstance(other, float) or isinstance(other, int):
            f = self.f + other
            grad = self.grad
            id = self.id
        else:
            f = self.f + other.f
            grad_aligned, other_grad_ligned = xa.align(
                self.grad, other.grad, join="outer", fill_value=0.0
            )
            grad = grad_aligned + other_grad_ligned
            id = grad.coords["a"].values.tolist()
        return Variable(id, f, grad)

    def __radd__(self, other: Self | float):
        return other + self

    def __mul__(self, other: Self):
        if isinstance(other, float) or isinstance(other, int):
            f = self.f * other
            grad = self.grad * other
            id = self.id
        else:
            f = self.f * other.f
            grad_aligned, other_grad_ligned = xa.align(
                self.grad, other.grad, join="outer", fill_value=0.0
            )
            grad = grad_aligned * other.f + other_grad_ligned * self.f
            id = grad.coords["a"].values.tolist()
        return Variable(id, f, grad)

    def __rmul__(self, other: Self | float):
        return self * other

    def __truediv__(self, other: Self | float):
        if isinstance(other, float) or isinstance(other, int):
            f = self.f / other
            grad = self.grad / other
            id = self.id
        else:
            f = self.f / other.f
            df, dg = xa.align(self.grad, other.grad, join="outer", fill_value=0.0)
            grad = (df * other.f - dg * self.f) / (other.f**2)
            id = grad.coords["a"].values.tolist()
        return Variable(id, f, grad)

    def __rtruediv__(self, other: Self | float):
        if isinstance(other, float) or isinstance(other, int):
            f = other / self.f
            grad = self.grad / self.f**2
            id = self.id
            return Variable(id, f, grad)
        return other.__truediv__(self)

    def __sub__(self, other: Self | float | int):
        return self + (-1) * other

    def __rsub__(self, other: Self | float | int):
        return other + (-1) * self

    def exp(self):
        f = np.exp(self.f)
        grad = np.exp(self.f) * self.grad
        return Variable(self.id, f, grad)

    def sin(self):
        f = np.sin(self.f)
        grad = np.cos(self.f) * self.grad
        return Variable(self.id, f, grad)

    def __pow__(self, n: float):
        f = self.f**n
        grad = n * self.f ** (n - 1) * self.grad
        return Variable(self.id, f, grad)

    def log(self):
        f = np.log(self.f)
        grad = self.grad / self.f
        return Variable(self.id, f, grad)

    def ndtr(self):
        f = ndtr(self.f)
        grad = norm.pdf(self.f) * self.grad
        return Variable(self.id, f, grad)


def main():
    pass


if __name__ == "__main__":
    main()
