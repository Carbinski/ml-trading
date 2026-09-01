import yfinance as yf

print("#" * 53)
print("#" * 18 + " YFINANCE LEARN " + "#" * 19)
print("#" * 53)

print("\nPRINTING MSFT DATA\n")

print("_" * 52)
print("_" * 19 + "MSFT" + "_" * 29)
print("_" * 52)

data = yf.Ticker("MSFT")

# Basic information about the stock
print("Info:")
print(data.info)   # Address, Industry/Sector, Summary, Company Officers, Most Recent Stock Data, etc.

# Calendar of the stock
print("Calendar:")
print(data.calendar)  # Dividend, Ex-Dividend Data, Earnings Data, Earnings High/Low/Avg, Rev High/Low/Avg

# Analyst price targets for the stock
print("Analyst Price Targets:")
print(data.analyst_price_targets)

# Quarterly income statement for the stock
print("Quarterly Income Statement:")
print(data.quarterly_income_stmt)  # EBITDA, Net Income, EPS, Total Revenue, etc.

# History of the stock
print("History:")
print(data.history(period='1mo'))  # Open, High, Low, Close, Volume, Dividends, Stock Splits

print("_" * 52)

# Storing multiple tickers
tickers = yf.Tickers('MSFT AAPL GOOG')  # Data is saved as pandas DataFrame
tickers.tickers['MSFT'].info
yf.download('MSFT AAPL GOOG', period='1mo')

# Funds
spy = yf.Ticker("SPY")

# Basic information about the fund
print("Info:")
print(spy.info)    # Address, Industry/Sector, Summary, Company Officers, Most Recent Stock Data, etc.

# Funds data for the fund
print("Funds Data:")
print(spy.funds_data)  # EBITDA, Net Income, EPS, Total Revenue, etc.

# Description of the fund
print("Description:")
print(spy.funds_data.description)  # Description of the fund

# Top holdings of the fund
print("Top Holdings:")
print(spy.funds_data.top_holdings)  # Top holdings of the fund
