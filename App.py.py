import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error

# Define rmse function (if it was used during model training)
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Load pre-trained models
@st.cache_resource
def load_model(file_name):
    try:
        return joblib.load(file_name)
    except FileNotFoundError:
        st.error(f"Model file {file_name} not found. Please ensure the file is in the correct location.")
        return None

models = {
    'Linear Regression': load_model('Linear_regresssion.pkl'),
    'Ridge Regression': load_model('Ridge_Regression.pkl'),
    'Lasso Regression': load_model('lasso_Regression.pkl'),
    'Random Forest': load_model('Random_Forest.pkl'),
    'Gradient Boost': load_model('Gradient_Boost.pkl')
}

# Define RMSE function for use in the app
def calculate_rmse(y_actual, y_pred):
    return np.sqrt(mean_squared_error(y_actual, y_pred))

# App Title
st.title("Predicting Sales Trends Using TATA online Retail datadet")
st.write("Select a model and input feature values to make predictions.")


Week = st.number_input("Week in a Year", min_value=1, max_value=52,step=1)
weekday = st.selectbox("Weekday", options=list(range(7)), format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x])
Day = st.number_input("Day in a Month", min_value=1, max_value=31, value=1)


# Convert inputs into a dataframe
input_data = pd.DataFrame({
    'Week': [Week],
    'Weekday': [weekday],
    'Day': [Day]
})

# Model Selection
available_models = {name: model for name, model in models.items() if model is not None}
if not available_models:
    st.error("No models available. Please check the model files.")
else:
    selected_model = st.selectbox("Choose a Model", options=list(available_models.keys()))
    st.write(f"You selected: {selected_model}")

    # Predict
    if st.button("Predict"):
        model = available_models[selected_model]
        try:
            prediction = model.predict(input_data)
            st.success(f"Predicted Quantity: {prediction[0]:.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

    # Display feature importances if applicable
if selected_model in ['Random Forest', 'Gradient Boost']:
    model = available_models[selected_model]
    try:
        # Access the best estimator if using RandomizedSearchCV
        if hasattr(model, 'best_estimator_'):
            model = model.best_estimator_
        
        # Access feature importances
        feature_importances = model.feature_importances_
        st.subheader("Feature Importances")
        
        # Ensure input_data is a DataFrame before accessing columns
        if isinstance(input_data, pd.Series):
            input_data = input_data.to_frame().T

        importance_df = pd.DataFrame({
            'Feature': input_data.columns,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        st.bar_chart(importance_df.set_index('Feature'))
    except AttributeError:
        st.warning("Feature importances not available for the selected model.")



# Footer
st.markdown("---")
st.markdown("Developed by Batch-2")
