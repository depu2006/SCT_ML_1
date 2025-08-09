import pandas as pd
import pickle as pk
import streamlit as st

st.header('🏠 Bangalore House Price Predictor')

# ✅ Load model and preprocessor
model, preprocessor = pk.load(open('C:/Users/91939/Downloads/HousePricePrediction/House_prediction_model.pkl', 'rb'))

# Load data just for location list
data = pd.read_csv('C:/Users/91939/Downloads/HousePricePrediction/Cleaned_data.csv')
if 'Unnamed: 0' in data.columns:
    data.drop(columns=['Unnamed: 0'], inplace=True)

# UI inputs
loc = st.selectbox('Choose Location', data['location'].unique())
sqft = st.number_input('Enter Total Sqft')
beds = st.number_input('Enter Number of Bedrooms')
bath = st.number_input('Enter Number of Bathrooms')
balc = st.number_input('Enter Number of Balconies')

# Create input DataFrame
input_df = pd.DataFrame([{
    'location': loc,
    'total_sqft': sqft,
    'bath': bath,
    'balcony': balc,
    'bedrooms': beds
}])

# ✅ Prediction
if st.button("Predict Price"):
    try:
        input_transformed = preprocessor.transform(input_df)
        prediction = model.predict(input_transformed)
        st.success(f"✅ Estimated House Price: ₹{prediction[0]*1e5:,.0f}")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
