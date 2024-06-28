import datetime as dt
from dataclasses import dataclass


@dataclass
class MarketData:
    bid: float
    ask: float
    date: dt.datetime
