import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Load dataset
dataset_path = "city_hour.csv"  # Ensure the file is in the correct location
df = pd.read_csv(dataset_path)

# Select relevant features
selected_features = ['PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
target = 'PM2.5'

# Remove missing values
df = df[selected_features + [target]].dropna()
print("Dataset Shape After Dropping NaN:", df.shape)  # Debugging

# Split dataset
X = df[selected_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save model and scaler
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

# Load model & scaler (avoids reloading on every request)
scaler = pickle.load(open("scaler.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return "Flask API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        print("Received Data:", data)  # Debugging

        # Convert to DataFrame
        input_data = pd.DataFrame([data])

        # Check for missing features
        if not all(feature in input_data.columns for feature in selected_features):
            return jsonify({"error": "Missing input features!"})

        # Preprocess input
        input_scaled = scaler.transform(input_data)
        print("Scaled Input:", input_scaled)  # Debugging

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        return jsonify({"prediction": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
