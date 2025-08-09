import os
import joblib
import numpy as np
from flask import Flask, request, jsonify

MODEL_DIR = "model"

# Load models & scalers
models = {
    "logistic": joblib.load(os.path.join(MODEL_DIR, "logistic_model.joblib")),
    "rf": joblib.load(os.path.join(MODEL_DIR, "rf_model.joblib")),
    "xgb": joblib.load(os.path.join(MODEL_DIR, "xgb_model.joblib"))
}

scalers = {
    "logistic": joblib.load(os.path.join(MODEL_DIR, "logistic_scaler.joblib")),
    "rf": joblib.load(os.path.join(MODEL_DIR, "rf_scaler.joblib")),
    "xgb": joblib.load(os.path.join(MODEL_DIR, "xgb_scaler.joblib"))
}

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

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    required_features = ["returns", "ma_10", "volatility", "rsi_14"]

    if not all(f in data for f in required_features):
        return jsonify({"error": "Missing features"}), 400

    features = np.array([data[f] for f in required_features]).reshape(1, -1)
    
    # Use XGBoost as main model here
    scaler = scalers["xgb"]
    model = models["xgb"]

    X_scaled = scaler.transform(features)
    proba = model.predict_proba(X_scaled)[0][1]
    pred = int(proba > 0.5)

    risk = classify_risk(data["volatility"])
    leverage = recommend_leverage(proba, risk)

    return jsonify({
        "direction": "Up" if pred else "Down",
        "confidence": round(proba, 4),
        "risk_level": risk,
        "recommended_leverage": leverage
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
