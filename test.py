import requests

url = "https://memcoin-server.onrender.com/predict"
sample_input = {
    "tx_idx_count": 123,
    "signing_wallet_nunique": 12,
    "quote_coin_amount_sum": 100.0,
    "quote_coin_amount_mean": 20.0,
    "base_coin_amount_sum": 300.0,
    "base_coin_amount_mean": 60.0,
    "buy_count": 4,
    "sell_count": 2,
    "buy_sell_ratio": 2.0,
    "decimals": 9,
    "bundle_size": 0
}

response = requests.post(url, json=sample_input)
print(response.json())
