import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset (Replace 'pollution_data.csv' with your actual file)
df = pd.read_csv('pollution_data.csv')

# Display basic info about the dataset
print(df.head())
print(df.info())

# Handling missing values
df = df.dropna()

# Selecting relevant features and target variable
features = ['Temperature', 'Humidity', 'Wind Speed', 'NO2', 'CO', 'SO2']  # Modify based on your dataset
target = 'PM2.5'

X = df[features]
y = df[target]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')

# Plotting actual vs predicted values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual PM2.5 Levels")
plt.ylabel("Predicted PM2.5 Levels")
plt.title("Actual vs Predicted PM2.5 Levels")
plt.show()