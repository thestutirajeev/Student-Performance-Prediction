# Python Backend (Flask)
# predict.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)  # 🔥 This enables CORS for all routes

# Load model parameters
params = joblib.load('custom_model_params.pkl')
weights = params['weights']
bias = params['bias']

# Basic scaler values from training data
# actual means and std from training data
mean = np.array([2.03544304, 0.33417722, 5.70886076, 3.10886076, 0.61265823, 0.12911392,
 3.55443038])
std = np.array([0.83817734, 0.74270905, 7.99295877, 1.11186807, 0.48714282, 0.33532599,
 1.3885424 ])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Extract features in correct order
    features = [
        data['studytime'],
        data['failures'],
        data['absences'],
        data['goout'],
        1 if data['famsup'] == 'yes' else 0,
        1 if data['schoolsup'] == 'yes' else 0,
        data['health']
    ]
    features = [float(f) for f in features]
    # Scale the input
    scaled = (np.array(features) - mean) / std

    # Predict
    prediction = np.dot(scaled, weights) + bias
    prediction = round(float(prediction), 2)

    return jsonify({'predicted_marks': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
