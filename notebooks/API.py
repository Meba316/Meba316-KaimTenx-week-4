import flask
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained models
rf_model = joblib.load('random_forest_model.pkl')  # Replace with your actual model filename
lstm_model = load_model('lstm_model.h5')  # Replace with your actual model filename

# Initialize a scaler for preprocessing if required
scaler = StandardScaler()

# Helper function to preprocess input data
def preprocess_input(data):
    # Assuming the input is a dictionary with keys matching the training data
    df = pd.DataFrame([data])  # Convert input data to a dataframe
    
    # Apply any feature engineering here as per Task 2's preprocessing (e.g., extracting weekdays, etc.)
    # Example: If we had a 'Date' field, we'd need to process it similarly.
    
    # For simplicity, let's assume the data matches the required input format already:
    return df

# Endpoint for predicting sales with the Random Forest model
@app.route('/predict_rf', methods=['POST'])
def predict_rf():
    try:
        # Extract JSON data from the request
        data = request.get_json()

        # Preprocess the input data
        input_data = preprocess_input(data)

        # Use the trained Random Forest model to make predictions
        prediction = rf_model.predict(input_data)

        # Return the prediction as JSON response
        return jsonify({'predicted_sales': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Endpoint for predicting sales with the LSTM model
@app.route('/predict_lstm', methods=['POST'])
def predict_lstm():
    try:
        # Extract JSON data from the request
        data = request.get_json()

        # Preprocess the input data (reshape for LSTM, scaling, etc.)
        input_data = preprocess_input(data)

        # Reshape the input for LSTM
        X_input = np.array(input_data).reshape((1, input_data.shape[0], input_data.shape[1]))

        # Use the trained LSTM model to make predictions
        prediction = lstm_model.predict(X_input)

        # Return the prediction as JSON response
        return jsonify({'predicted_sales': prediction[0][0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Root endpoint to check if the server is running
@app.route('/')
def home():
    return "Model API is Running!"

if __name__ == '__main__':
    # Run the API on a local server (use host='0.0.0.0' to allow access from outside the container if needed)
    app.run(debug=True, host='0.0.0.0', port=5000)
