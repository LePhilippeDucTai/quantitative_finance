from typing import Self

import numpy as np
import xarray as xa


def make_gradient(grad: float, id: str | list):
    return xa.DataArray([grad], coords={"a": list(id)})


class Variable:
    def __init__(self, id: str, f: float = 1.0, grad: float = 1.0):
        self.id = [id] if isinstance(id, str) else id
        self.f = f
        self.grad = make_gradient(grad, id) if isinstance(grad, float) else grad

    def __add__(self, other: Self):
        f = self.f + other.f
        grad_aligned, other_grad_ligned = xa.align(
            self.grad, other.grad, join="outer", fill_value=0.0
        )
        df = grad_aligned + other_grad_ligned
        id = df.coords["a"].values.tolist()
        return Variable(id, f, df)

    def __mul__(self, other: Self):
        f = self.f * other.f
        grad_aligned, other_grad_ligned = xa.align(
            self.grad, other.grad, join="outer", fill_value=0.0
        )
        grad = grad_aligned * other.f + other_grad_ligned * self.f
        id = grad.coords["a"].values.tolist()
        return Variable(id, f, grad)

    def exp(self):
        f = np.exp(self.f)
        grad = np.exp(self.f) * self.grad
        return Variable(self.id, f, grad)

    def sin(self):
        f = np.sin(self.f)
        grad = np.cos(self.f) * self.grad
        return Variable(self.id, f, grad)

    def __repr__(self):
        return f"Var[{self.id}](f={self.f}, grad={self.grad.data})"

    def __pow__(self, n: float):
        f = self.f**n
        grad = n * self.f ** (n - 1) * self.grad
        return Variable(self.id, f, grad)

    def log(self):
        f = np.log(self.f)
        grad = self.grad / self.f
        return Variable(self.id, f, grad)


def main():
    x = Variable("x", f=3)
    y = Variable("y", f=-1)
    z = Variable("z", f=2)
    f = (((x * y) * (y * z).sin()) ** 2).log()
    print(f)


if __name__ == "__main__":
    main()
