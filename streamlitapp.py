import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved model
Breast_Cancer_project = pickle.load(open('Breast_Cancer_model.sav', 'rb'))

# page title
st.title('Breast Cancer Prediction using ML')

# input columns
col1, col2, col3 = st.columns(3)

with col1:
    diagnosis = st.selectbox('Diagnosis (M=1, B=0)', [0, 1], index=0)
    radius_mean = st.number_input('Radius Mean', min_value=0.0, value=17.99, step=0.01)
    texture_mean = st.number_input('Texture Mean', min_value=0.0, value=10.38, step=0.01)

with col2:
    smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, value=0.1184, step=0.0001)
    compactness_mean = st.number_input('Compactness Mean', min_value=0.0, value=0.2776, step=0.0001)
    concavity_mean = st.number_input('Concavity Mean', min_value=0.0, value=0.3001, step=0.0001)

with col3:
    concave_points_mean = st.number_input('Concave Points Mean', min_value=0.0, value=0.1471, step=0.0001)
    symmetry_mean = st.number_input('Symmetry Mean', min_value=0.0, value=0.2419, step=0.0001)
    fractal_dimension_mean = st.number_input('Fractal Dimension Mean', min_value=0.0, value=0.07871, step=0.00001)

# code for prediction
Breast_Cancer_diagnosis = ''

# creating a button for prediction
if st.button('🔍 Breast Cancer Test'):
    try:
        features = [[
            diagnosis, radius_mean, texture_mean, perimeter_mean, area_mean,
            smoothness_mean, compactness_mean, concavity_mean,
            concave_points_mean, symmetry_mean, fractal_dimension_mean
        ]]
        Breast_Cancer_prediction = Breast_Cancer_project.predict(features)
    except ValueError as e:
        st.error(f"Prediction error: {str(e)}")

    if Breast_Cancer_prediction[0] == 1:
        Breast_Cancer_diagnosis = 'The person has Malignant Breast Cancer 😷'
    else:
        Breast_Cancer_diagnosis = 'The person has Benign Breast Cancer 😊'

st.success(Breast_Cancer_diagnosis)
