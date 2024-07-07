from typing import Self

import numpy as np


class Variable:
    def __init__(self, f: float | np.ndarray, df: np.ndarray | None = None):
        self.f = np.array(f)
        self.df = df if df is not None else np.ones_like(f)

    def __pow__(self, a: float) -> Self:
        x = self.f**a
        dx = self.df * a * self.f ** (a - 1)
        return Variable(x, dx)

    def __add__(self, other: Self | float | np.ndarray) -> Self:
        match other:
            case Variable(f=y, df=dy):
                x = self.f + y
                dx = self.df + dy
            case _:
                x = self.f + other
                dx = self.df
        return Variable(x, dx)

    def __radd__(self, other: Self) -> Self:
        return self.__add__(other)

    def __sub__(self, other: Self) -> Self:
        return self.__add__(-1 * other)

    def __mul__(self, other: Self | float | np.ndarray) -> Self:
        match other:
            case Variable(f=y, df=dy):
                x = self.f * y
                dx = self.df * y + self.f * dy
            case _:
                x = self.f * other
                dx = self.df * other
        return Variable(x, dx)

    def __rmul__(self, other: Self):
        return self.__mul__(other)

    def __truediv__(self, other: Self):
        match other:
            case Variable(f=g, df=dg):
                x = self.f / g
                dx = self.df / g - self.f / (g * g) * dg
            case _:
                return self * (1 / other)
        return Variable(x, dx)

    def __rtruediv__(self, other: float | np.ndarray):
        match other:
            case Variable(_, _):
                return other / self
            case _:
                x = other / self.f
                dx = -other / (self.f * self.f)
                return Variable(x, dx)

    def exp(self) -> Self:
        x = np.exp(self.f)
        dx = np.exp(self.f) * self.df
        return Variable(x, dx)

    def __repr__(self):
        return f"Variable(f={self.f}, df={self.df})"

    def sin(self) -> Self:
        x = np.sin(self.f)
        dx = np.cos(self.f) * self.df
        return Variable(x, dx)

    def log(self) -> Self:
        x = np.log(self.f)
        dx = self.df * (1 / self.f)
        return Variable(x, dx)


def main():
    # x = Variable(np.array([3])[:, np.newaxis, np.newaxis])
    # y = Variable(np.array([-1])[np.newaxis, :, np.newaxis])
    # z = Variable(np.array([2])[np.newaxis, np.newaxis, :])
    # f = x * y * (z * y).sin()
    # print(f)
    x = Variable([1, 2, 3])
    a = Variable(1)
    z = x.exp() ** 2
    y = ((x.exp() + a).sin() * z).exp() + a
    print(y)


if __name__ == "__main__":
    main()
