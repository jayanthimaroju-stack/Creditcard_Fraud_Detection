from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Allow the frontend to communicate with Flask
CORS(app)

# Get the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the trained model and scaler
model_path = os.path.join(BASE_DIR, "models", "fraud_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Expected order of the 30 input features
FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Credit Card Fraud Detection API is running!",
        "status": "success"
    })


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Send a JSON object containing features."
        }), 400

    features = data.get("features")

    if not isinstance(features, list) or len(features) != 30:
        return jsonify({
            "error": "Please provide exactly 30 numeric features."
        }), 400

    try:
        transaction = pd.DataFrame([features], columns=FEATURES)
        transaction = transaction.astype(float)

        # Reject missing or infinite values
        if not transaction.map(lambda x: pd.notna(x) and abs(x) != float("inf")).all().all():
            return jsonify({
                "error": "Features must contain only finite numeric values."
            }), 400

    except (ValueError, TypeError):
        return jsonify({
            "error": "All 30 features must be numeric."
        }), 400

    try:
        scaled_transaction = scaler.transform(transaction)

        prediction = int(model.predict(scaled_transaction)[0])
        probabilities = model.predict_proba(scaled_transaction)[0]

        fraud_index = list(model.classes_).index(1)
        fraud_probability = float(probabilities[fraud_index])

        return jsonify({
            "prediction": prediction,
            "result": "FRAUDULENT" if prediction == 1 else "LEGITIMATE",
            "fraud_probability": round(fraud_probability * 100, 2)
        })

    except Exception:
        app.logger.exception("Transaction prediction failed")
        return jsonify({
            "error": "Prediction failed. Check the model and scaler configuration."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)