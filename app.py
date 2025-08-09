import numpy as np
import joblib
from flask import Flask, request, jsonify
import os

MODEL_DIR = "model"

app = Flask(__name__)

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

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    required_features = ["returns", "ma_10", "volatility", "rsi_14"]
    if not all(k in data for k in required_features):
        return jsonify({"error": "Missing required features"}), 400

    X_input = np.array([[data[f] for f in required_features]])
    try:
        model = joblib.load(f"{MODEL_DIR}/xgboost_model.joblib")
        scaler = joblib.load(f"{MODEL_DIR}/xgboost_scaler.joblib")
        X_scaled = scaler.transform(X_input)
        prob = model.predict_proba(X_scaled)[0][1]
        pred = int(prob > 0.5)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    risk = classify_risk(data["volatility"])
    leverage = recommend_leverage(prob, risk)

    return jsonify({
        "direction": "Up" if pred else "Down",
        "confidence": round(prob, 4),
        "risk_level": risk,
        "recommended_leverage": leverage
    })

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "running"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
