import yfinance as yf


def fetch_market_data(symbol: str):
    yf.Ticker(symbol)


def main():
    tick = yf.Ticker("AAPL")
    print(tick.shares)
    print(dir(tick))


if __name__ == "__main__":
    main()
