import requests
import pandas as pd

API_URL = "https://api.coingecko.com/api/v3"

def float_to_hex(f):
    _, hex_representation = f.hex().split('x')
    return "0x" + hex_representation

def get_new_tokens():
    response = requests.get(f"{API_URL}/coins/ethereum/market_chart", params={"vs_currency": "usd", "days": 1})
    data = response.json()
    
    new_tokens = []
    for token in data["market_caps"]:
        timestamp, market_cap = token
        if market_cap > 20000:
            coin_token = data["prices"][data["market_caps"].index(token)][1]
            coin_token_hex = float_to_hex(coin_token)
            new_tokens.append((coin_token_hex, timestamp, market_cap))
    return new_tokens

if __name__ == "__main__":
    new_tokens = get_new_tokens()
    if new_tokens:
        print("New Ethereum Tokens Created in the Last 24 Hours (Market Cap > $20,000)")
        for coin_token, timestamp, market_cap in new_tokens:
            time_created = pd.to_datetime(timestamp, unit="ms").strftime("%H:%M:%S")
            print(f"COIN: {coin_token}, TIME CREATED: {time_created}, MARKET CAP: ${market_cap/1e6:.1f}MM")
    else:
        print("No new Ethereum tokens with market cap > $20,000 created in the last 24 hours.")
