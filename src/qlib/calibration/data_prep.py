from dataclasses import dataclass

import pandas as pd
import yfinance as yf


@dataclass
class MarketData:
    spot_price: float
    call_prices: pd.DataFrame


def call_data(ticker: yf.Ticker, expiry_date: str) -> pd.DataFrame:
    return (
        ticker.option_chain(date=expiry_date)
        .calls[["strike", "lastPrice"]]
        .assign(expiry_date=expiry_date)
    )


def get_spot_price(ticker: yf.Ticker):
    return ticker.history(period="1d")["Close"].iloc[0]


def fetch_data(symbol: str):
    tick = yf.Ticker(symbol)
    expiry_dates = tick.options
    data_calls = pd.concat([call_data(tick, exp) for exp in expiry_dates])
    spot = get_spot_price(tick)
    return MarketData(spot_price=spot, call_prices=data_calls)
