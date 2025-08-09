import os
import pandas as pd
import numpy as np
import requests
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime, timedelta

# Configuration
FASTFOREX_API_KEY = os.getenv("FASTFOREX_API_KEY")
FASTFOREX_BASE_URL = "https://api.fastforex.io"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Fetch historical EURUSD data
def fetch_forex_data(symbol="EURUSD"):
    base = symbol[:3]
    target = symbol[3:]
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=30)  # last 30 days
    
    all_data = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        url = f"{FASTFOREX_BASE_URL}/historical"
        params = {
            "date": date_str,
            "from": base,
            "to": target,
            "api_key": FASTFOREX_API_KEY
        }
        resp = requests.get(url, params=params)
        if resp.status_code == 200:
            res = resp.json().get("results")
            if res and target in res:
                all_data.append({"date": date_str, "close": res[target]})
        current_date += timedelta(days=1)
    
    df = pd.DataFrame(all_data)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.resample('1h').ffill()  # hourly frequency with forward fill
    return df

# Feature engineering
def compute_rsi(series: pd.Series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def add_features(df):
    df["returns"] = df["close"].pct_change()
    df["ma_10"] = df["close"].rolling(window=10).mean()
    df["volatility"] = df["returns"].rolling(window=10).std()
    df["rsi_14"] = compute_rsi(df["close"])
    df.dropna(inplace=True)
    df["direction"] = (df["returns"].shift(-1) > 0).astype(int)
    return df

# Models
class ForexDataset(Dataset):
    def __init__(self, data, target, seq_len=10):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(target[i+seq_len])
        self.X = torch.tensor(np.array(X), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

# Training function
def train_models(df):
    features = ["returns", "ma_10", "volatility", "rsi_14"]
    X = df[features].values
    y = df["direction"].values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Logistic Regression
    scaler_lr = StandardScaler()
    X_train_lr = scaler_lr.fit_transform(X_train)
    model_lr = LogisticRegression(max_iter=1000)
    model_lr.fit(X_train_lr, y_train)
    joblib.dump(model_lr, os.path.join(MODEL_DIR, "logistic_model.joblib"))
    joblib.dump(scaler_lr, os.path.join(MODEL_DIR, "logistic_scaler.joblib"))

    # Random Forest
    scaler_rf = StandardScaler()
    X_train_rf = scaler_rf.fit_transform(X_train)
    model_rf = RandomForestClassifier(n_estimators=100)
    model_rf.fit(X_train_rf, y_train)
    joblib.dump(model_rf, os.path.join(MODEL_DIR, "rf_model.joblib"))
    joblib.dump(scaler_rf, os.path.join(MODEL_DIR, "rf_scaler.joblib"))

    # XGBoost
    scaler_xgb = StandardScaler()
    X_train_xgb = scaler_xgb.fit_transform(X_train)
    model_xgb = XGBClassifier(eval_metric='logloss')
    model_xgb.fit(X_train_xgb, y_train)
    joblib.dump(model_xgb, os.path.join(MODEL_DIR, "xgb_model.joblib"))
    joblib.dump(scaler_xgb, os.path.join(MODEL_DIR, "xgb_scaler.joblib"))

    # LSTM
    scaler_lstm = MinMaxScaler()
    X_norm = scaler_lstm.fit_transform(X)
    seq_len = 10

    train_dataset = ForexDataset(X_norm[:len(X_train)], y_train, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    lstm_model = LSTMModel(input_size=len(features))
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    lstm_model.train()
    for epoch in range(10):
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = lstm_model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

    torch.save(lstm_model.state_dict(), os.path.join(MODEL_DIR, "lstm_model.pth"))
    joblib.dump(scaler_lstm, os.path.join(MODEL_DIR, "lstm_scaler.joblib"))

# Risk and leverage functions
def classify_risk(volatility):
    if volatility > 0.015:
        return "High"
    elif volatility > 0.007:
        return "Medium"
    else:
        return "Low"

def recommend_leverage(confidence, risk):
    if confidence > 0.85 and risk == "Low":
        return 100
    elif confidence > 0.7 and risk != "High":
        return 50
    elif confidence > 0.5:
        return 20
    else:
        return 5

def main():
    print("Fetching data...")
    df_raw = fetch_forex_data()
    df = add_features(df_raw)
    if df.empty:
        print("No data to train on. Exiting.")
        return
    print("Training models...")
    train_models(df)
    print("Training complete.")

if __name__ == "__main__":
    main()
