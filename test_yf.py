import yfinance as yf
try:
    data = yf.download("BTC-USD", period="30d", interval="1m")
    print(len(data))
except Exception as e:
    print("Error:", e)
