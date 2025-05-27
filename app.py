# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json

app = Flask(__name__)

model = joblib.load('model.pkl')
with open("features.json") as f:
    features = json.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    df = df.reindex(columns=features, fill_value=0)
    prediction = model.predict_proba(df)[:, 1]
    return jsonify({'probability': prediction[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
