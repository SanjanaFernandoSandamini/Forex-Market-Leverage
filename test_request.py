import requests

url = "http://localhost:5000/predict"
payload = {
    "returns": 0.0005,
    "ma_10": 1.105,
    "volatility": 0.008,
    "rsi_14": 55.0
}

response = requests.post(url, json=payload)
print(response.json())
