import requests
import json

# Замініть на ваш URL з Render
RENDER_URL = "https://memcoin-server.onrender.com"

# Тест здоров'я
try:
    response = requests.get(f"{RENDER_URL}/health")
    print("Health check:", response.json())
except Exception as e:
    print(f"Health check failed: {e}")

# Тест прогнозування
test_data = {
    "tx_idx_count": 150,
    "signing_wallet_nunique": 45,
    "quote_coin_amount_sum": 1000.5,
    "quote_coin_amount_mean": 6.67,
    "base_coin_amount_sum": 50000,
    "base_coin_amount_mean": 333.33,
    "buy_count": 80,
    "sell_count": 20,
    "bundle_size": 1,
    "decimals": 6
}

try:
    response = requests.post(
        f"{RENDER_URL}/predict",
        json=test_data,
        headers={'Content-Type': 'application/json'}
    )
    print("Prediction:", response.json())
except Exception as e:
    print(f"Prediction failed: {e}")