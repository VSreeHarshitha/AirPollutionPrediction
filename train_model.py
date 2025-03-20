import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Define dataset folder path
folder_path = r"C:\Users\Sree harshitha\OneDrive\Desktop\air pollution pred"

# Find the dataset file
file_path = os.path.join(folder_path, "city_hour.csv")  # Ensure this is the correct file

# Load dataset
df = pd.read_csv(file_path)

# Print available columns
print("✅ Dataset Loaded Successfully!")
print(f"🔹 Available columns: {df.columns.tolist()}")

# Selecting relevant features (excluding missing columns)
features = ['PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene']
target = 'PM2.5'  # Predicting PM2.5 levels

# Check if selected columns exist
missing_cols = [col for col in features if col not in df.columns]
if missing_cols:
    print(f"❌ ERROR: Missing columns in dataset: {missing_cols}")
    exit()  # Stop execution if columns are missing

# Drop rows with missing values
df = df[features + [target]].dropna()

# Split dataset into features (X) and target variable (y)
X = df[features]
y = df[target]

# Split into train & test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save model & scaler
pickle.dump(model, open(os.path.join(folder_path, "model.pkl"), "wb"))
pickle.dump(scaler, open(os.path.join(folder_path, "scaler.pkl"), "wb"))

print("✅ Model trained and saved successfully!")
print(df.head())  # Show first 5 rows
