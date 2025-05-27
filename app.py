from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)

# Завантажити модель при старті
try:
    model = joblib.load('memcoin_model.pkl')
    with open('feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = []


def clean_feature_names(df):
    df = df.copy()
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)
    df.columns = df.columns.str.replace('__+', '_', regex=True)
    df.columns = df.columns.str.strip('_')
    return df


@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Memcoin Graduation Predictor',
        'status': 'running',
        'model_loaded': model is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        df = pd.DataFrame([data])

        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0

        X = df[feature_names].fillna(0)

        X = clean_feature_names(X)

        prediction = model.predict_proba(X)[0][1]

        return jsonify({
            'graduation_probability': float(np.clip(prediction, 0.0001, 0.9999)),
            'features_used': len(feature_names)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)