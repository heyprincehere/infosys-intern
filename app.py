import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer

# Load pre-trained models
models = {
    "Random Forest": "Randomforest_rfm.pkl",
    "Support Vector Regression": "svr_model.pkl"
}

# Load scaler function
@st.cache_resource
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = joblib.load(f)
    return scaler

# Preprocessing pipeline
def create_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            ('frequency_cluster_recency', 'passthrough', ['Frequency', 'Cluster', 'Recency'])
        ]
    )
    return preprocessor

# Load model function
@st.cache_resource
def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    return model

# Title and Introduction
st.title("Sales trends Prediction Application")
st.write("This application predicts the Monetary value based on Frequency, Cluster, and Recency.")

# Inputs
frequency = st.number_input("Enter Frequency", min_value=0, value=1)
cluster = st.selectbox("Enter Cluster", [0,1,2,3])  # Assuming cluster is an integer
recency = st.number_input("Enter Recency", min_value=0, value=1)

# Model selection
selected_model = st.selectbox("Choose a Model", list(models.keys()))

# Process inputs
if st.button("Predict"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        "Frequency": [frequency],
        "Cluster": [cluster],
        "Recency": [recency]
    })

    # Load preprocessor, model, and scaler
    preprocessor = create_preprocessor()  # Create a new preprocessor instance
    model_path = models[selected_model]  # Get the model path from the dictionary
    model = load_model(model_path)  # Load the model using the path
    scaler = load_scaler('y_scaler.pkl')  # Path to your saved scaler

    # Fit the preprocessor on the input data (or training data if available)
    preprocessor.fit(input_data)  # Fit the preprocessor on the input data

    # Preprocess inputs
    X_processed = preprocessor.transform(input_data)

    # Predict
    prediction = model.predict(X_processed)

    # Rescale the prediction
    rescaled_prediction = scaler.inverse_transform(prediction.reshape(-1, 1))

    # Display results
    st.write(f"Predicted Sales Value: {float(rescaled_prediction[0]):.2f}")