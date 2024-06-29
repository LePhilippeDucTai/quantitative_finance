"""Generic library to calculate Greeks : Delta, Gamma, Vega, Rho."""

from qlib.traits import ItoProcess


class Greeks:
    def __init__(self, model: ItoProcess, payoff_func: callable):
        self.model = model
        self.payoff_func = payoff_func

    def delta(self):
        return
