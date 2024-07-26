import datetime as dt
from dataclasses import dataclass
from enum import Enum, auto


class Action(Enum):
    BUY = auto()
    SELL = auto()


@dataclass
class Quote:
    price: float
    quantity: int


@dataclass
class MarketData:
    ticker: str
    bid: Quote
    ask: Quote


@dataclass
class Order:
    ticker: str
    quantity: int
    timestamp: dt.datetime
    action: Action
