import pickle
import streamlit as st

# loading the saved model
Breast_Cancer_project = pickle.load(open('Breast_Cancer_model.sav', 'rb'))

# page title
st.title('Breast Cancer Prediction using ML')

# input columns
col1, col2, col3 = st.columns(3)

# Inputs for 30 features
with col1:
    diagnosis = st.selectbox('Diagnosis (M=1, B=0)', [0, 1], index=0)
    radius_mean = st.number_input('Radius Mean', min_value=0.0, value=17.99, step=0.01)
    texture_mean = st.number_input('Texture Mean', min_value=0.0, value=10.38, step=0.01)
    perimeter_mean = st.number_input('Perimeter Mean', min_value=0.0, value=122.8, step=0.1)
    area_mean = st.number_input('Area Mean', min_value=0.0, value=1001.0, step=1.0)
    smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, value=0.1184, step=0.0001)
    compactness_mean = st.number_input('Compactness Mean', min_value=0.0, value=0.2776, step=0.0001)
    concavity_mean = st.number_input('Concavity Mean', min_value=0.0, value=0.3001, step=0.0001)
    concave_points_mean = st.number_input('Concave Points Mean', min_value=0.0, value=0.1471, step=0.0001)
    symmetry_mean = st.number_input('Symmetry Mean', min_value=0.0, value=0.2419, step=0.0001)

with col2:
    fractal_dimension_mean = st.number_input('Fractal Dimension Mean', min_value=0.0, value=0.07871, step=0.00001)
    radius_se = st.number_input('Radius SE', min_value=0.0, value=1.095, step=0.01)
    texture_se = st.number_input('Texture SE', min_value=0.0, value=0.9053, step=0.01)
    perimeter_se = st.number_input('Perimeter SE', min_value=0.0, value=8.589, step=0.1)
    area_se = st.number_input('Area SE', min_value=0.0, value=153.4, step=1.0)
    smoothness_se = st.number_input('Smoothness SE', min_value=0.0, value=0.006399, step=0.0001)
    compactness_se = st.number_input('Compactness SE', min_value=0.0, value=0.04904, step=0.0001)
    concavity_se = st.number_input('Concavity SE', min_value=0.0, value=0.05373, step=0.0001)
    concave_points_se = st.number_input('Concave Points SE', min_value=0.0, value=0.01587, step=0.0001)
    symmetry_se = st.number_input('Symmetry SE', min_value=0.0, value=0.03003, step=0.0001)

with col3:
    fractal_dimension_se = st.number_input('Fractal Dimension SE', min_value=0.0, value=0.006193, step=0.00001)
    radius_worst = st.number_input('Radius Worst', min_value=0.0, value=25.38, step=0.01)
    texture_worst = st.number_input('Texture Worst', min_value=0.0, value=17.33, step=0.01)
    perimeter_worst = st.number_input('Perimeter Worst', min_value=0.0, value=184.6, step=0.1)
    area_worst = st.number_input('Area Worst', min_value=0.0, value=2019.0, step=1.0)
    smoothness_worst = st.number_input('Smoothness Worst', min_value=0.0, value=0.1622, step=0.0001)
    compactness_worst = st.number_input('Compactness Worst', min_value=0.0, value=0.6656, step=0.0001)
    concavity_worst = st.number_input('Concavity Worst', min_value=0.0, value=0.7119, step=0.0001)
    concave_points_worst = st.number_input('Concave Points Worst', min_value=0.0, value=0.2654, step=0.0001)
    symmetry_worst = st.number_input('Symmetry Worst', min_value=0.0, value=0.4601, step=0.0001)
    fractal_dimension_worst = st.number_input('Fractal Dimension Worst', min_value=0.0, value=0.1189, step=0.00001)

# code for prediction
Breast_Cancer_diagnosis = ''

# creating a button for prediction
if st.button('🔍 Breast Cancer Test'):
    try:
        features = [[
            diagnosis, radius_mean, texture_mean, perimeter_mean, area_mean,
            smoothness_mean, compactness_mean, concavity_mean, concave_points_mean,
            symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se,
            area_se, smoothness_se, compactness_se, concavity_se, concave_points_se,
            symmetry_se, fractal_dimension_se, radius_worst, texture_worst,
            perimeter_worst, area_worst, smoothness_worst, compactness_worst,
            concavity_worst, concave_points_worst, symmetry_worst,
            fractal_dimension_worst
        ]]
        Breast_Cancer_prediction = Breast_Cancer_project.predict(features)
    except ValueError as e:
        st.error(f"Prediction error: {str(e)}")

    if Breast_Cancer_prediction[0] == 1:
        Breast_Cancer_diagnosis = 'The person has Malignant Breast Cancer 😷'
    else:
        Breast_Cancer_diagnosis = 'The person has Benign Breast Cancer 😊'

st.success(Breast_Cancer_diagnosis)
