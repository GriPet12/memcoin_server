from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load('memcoin_graduation_model.pkl')

with open('features.txt', 'r') as f:
    features = [line.strip() for line in f.readlines()]


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        df = pd.DataFrame([data])

        for feature in features:
            if feature not in df.columns:
                df[feature] = 0

        X = df[features].fillna(0)

        prediction = model.predict_proba(X)[0][1]

        return jsonify({
            'graduation_probability': float(np.clip(prediction, 0.0001, 0.9999))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)