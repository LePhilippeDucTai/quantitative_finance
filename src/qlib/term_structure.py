import datetime as dt
from abc import ABC, abstractmethod

import numpy as np


class DayCountConvention:
    Actual360 = 1 / 360
    Actual365 = 1 / 365
    One = 1


class RiskFreeCurve(ABC):

    @abstractmethod
    def instantaneous(self, t: float):
        return


class FlatForward(RiskFreeCurve):
    def __init__(self, r: float, scale: DayCountConvention):
        self.r = r
        self.scale = scale

    def instantaneous(self, _: float):
        return self.r * self.scale


class FlatTermStructure:
    def __init__(self, risk_free_curve: FlatForward):
        self.risk_free_curve = risk_free_curve

    def discount_factor(self, time_to_maturity):
        return np.exp(-self.risk_free_curve.instantaneous(0) * time_to_maturity)
