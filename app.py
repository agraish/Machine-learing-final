
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('best_xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')

st.title('Prediksi Harga Berlian')
st.write('Aplikasi sederhana untuk memprediksi Price(in US dollars)')

input_data = pd.DataFrame(0, index=[0], columns=feature_columns)

for col in feature_columns:
    if col.startswith('cut_') or col.startswith('color_') or col.startswith('clarity_'):
        continue
    input_data.loc[0, col] = st.number_input(col, value=1.0)

if st.button('Prediksi'):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f'Prediksi Price(in US dollars): {prediction[0]:.2f}')
