iimport pandas as pd
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime, timedelta
import os

# CONFIGURATION
fastforex_api_key = os.environ.get("FASTFOREX_API_KEY")
if not fastforex_api_key:
    raise Exception("FASTFOREX_API_KEY not set in environment")

FASTFOREX_BASE_URL = "https://api.fastforex.io"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# DATA FETCHING
def fetch_forex_data(symbol="EURUSD"):
    base_currency = symbol[:3]
    target_currency = symbol[3:]
    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=13)

    all_data = []
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        url = f"{FASTFOREX_BASE_URL}/historical"
        params = {"date": date_str, "from": base_currency, "to": target_currency, "api_key": fastforex_api_key}
        response = requests.get(url, params=params)
        if response.status_code != 200:
            current_date += timedelta(days=1)
            continue
        json_data = response.json().get("results")
        if json_data and target_currency in json_data:
            all_data.append({"t": date_str, "close": json_data[target_currency]})
        current_date += timedelta(days=1)

    df = pd.DataFrame(all_data)
    df["t"] = pd.to_datetime(df["t"])
    df.set_index("t", inplace=True)
    df = df.resample('1h').ffill()
    return df

# FEATURE ENGINEERING
def compute_rsi(series: pd.Series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def add_features(df):
    if df.empty or len(df) < 14:
        print("Insufficient data for feature engineering.")
        return df
    df["returns"] = df["close"].pct_change()
    df["ma_10"] = df["close"].rolling(10).mean()
    df["volatility"] = df["returns"].rolling(10).std()
    df["rsi_14"] = compute_rsi(df["close"])
    df.dropna(inplace=True)
    df["direction"] = (df["returns"].shift(-1) > 0).astype(int)
    return df

# MODEL EVALUATION
def evaluate_model(model, scaler, X_test, y_test):
    X_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_scaled)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return acc, prec, rec, f1

# TRAINING FUNCTION
def train_models(df):
    features = ["returns", "ma_10", "volatility", "rsi_14"]
    X = df[features]
    y = df["direction"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100),
        "xgboost": XGBClassifier(eval_metric='logloss')
    }

    scalers = {}

    for name, model in models.items():
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
        joblib.dump(model, f"{MODEL_DIR}/{name}_model.joblib")
        joblib.dump(scaler, f"{MODEL_DIR}/{name}_scaler.joblib")
        scalers[name] = scaler

    # Evaluation
    for name, model in models.items():
        scaler = scalers[name]
        acc, prec, rec, f1 = evaluate_model(model, scaler, X_test, y_test)
        print(f"{name} - Acc: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # LSTM model training
    mm = MinMaxScaler()
    X_norm = mm.fit_transform(X)
    seq_len = 10

    class ForexDataset(Dataset):
        def __init__(self, data, target):
            X_list, y_list = [], []
            for i in range(len(data) - seq_len):
                X_list.append(data[i:i+seq_len])
                y_list.append(target[i+seq_len])
            self.X = torch.tensor(np.array(X_list), dtype=torch.float32)
            self.y = torch.tensor(np.array(y_list), dtype=torch.long)
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

    dataset = ForexDataset(X_norm, y.values)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    lstm_model = LSTMModel(len(features))
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = lstm_model(xb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

    torch.save(lstm_model.state_dict(), f"{MODEL_DIR}/lstm_model.pth")
    joblib.dump(mm, f"{MODEL_DIR}/lstm_scaler.joblib")

if __name__ == "__main__":
    print("Fetching data...")
    df_raw = fetch_forex_data("EURUSD")
    df_feat = add_features(df_raw)
    if df_feat.empty:
        print("No data to train on.")
    else:
        train_models(df_feat)
