# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

# Load your dataset
@st.cache
def load_data():
    df = pd.read_csv(r"C:\Users\sairi\.cache\kagglehub\datasets\ishanshrivastava28\tata-online-retail-dataset\versions\1/Online Retail Data Set.csv", encoding='ISO-8859-1')
    return df

# Load data
df = load_data()

# Preprocess the data (similar to your existing code)
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['Country'] = df['Country'].astype('category').cat.codes
# Extract time-based features
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d-%m-%Y %H:%M')

# Extracting weekday name
df['WeekDay'] = df['InvoiceDate'].dt.strftime('%a')  # Ensure this line is executed
df['Month'] = df['InvoiceDate'].dt.month
df['Day'] = df['InvoiceDate'].dt.day
# Now you can convert 'WeekDay' to categorical
df['WeekDay'] = df['WeekDay'].astype('category').cat.codes
df['Month'] = df['Month'].astype('category').cat.codes
df['Day'] = df['Day'].astype('category').cat.codes

# Define features and target variable
features = ['Quantity', 'UnitPrice', 'Month', 'Day', 'WeekDay']
target = 'TotalAmount'
X = df[features]
y = df[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Streamlit UI
st.title("TATA Online Retail Sales Prediction")
st.write("### Model Evaluation Metrics")
st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# User input for prediction
st.sidebar.header("User  Input Features")
quantity = st.sidebar.number_input("Quantity", min_value=1, value=1)
unit_price = st.sidebar.number_input("Unit Price", min_value=0.01, value=1.0)
month = st.sidebar.number_input("Month (1-12)", min_value=1, max_value=12, value=1)
day = st.sidebar.number_input("Day (1-31)", min_value=1, max_value=31, value=1)
weekday = st.sidebar.number_input("Weekday (0-6)", min_value=0, max_value=6, value=0)

# Create a DataFrame for the input
input_data = pd.DataFrame([[quantity, unit_price, month, day, weekday]], columns=features)

# Make prediction
if st.sidebar.button("Predict"):
    prediction = rf_model.predict(input_data)
    st.write(f"Predicted Total Amount: ${prediction[0]:.2f}")

# Optional: Add more visualizations or features as needed