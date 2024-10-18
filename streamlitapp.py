import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# loading the saved model
Breast_Cancer_project = pickle.load(open('Breast_Cancer_model.sav', 'rb'))

# page title
st.title('Breast Cancer Prediction using ML')

# Inject custom CSS
st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background-image: url("https://raw.githubusercontent.com/SHAIK-RAIYAN-2022-CSE/malaria/main/Images-free-abstract-minimalist-wallpaper-HD.jpg");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }
        .block-container {
            background: rgba(255, 255, 255, 0.1);
            padding: 30px;
            border: 2px solid #ccc;
            border-radius: 15px;
            max-width: 950px;
            margin: auto;
            backdrop-filter: blur(10px);
            box-shadow: 0px 8px 30px rgba(0, 0, 0, 0.7);
        }
        h1 {
            color: #ccc;
            text-align: center;
            margin-bottom: 20px;
        }
        .stButton>button {
            background-color: #FF6347;
            color: white;
            font-size: 20px;
            padding: 12px 30px;
            border-radius: 10px;
            border: none;
            transition: 0.3s;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .stButton>button:hover {
            background-color: white;
            color: #FF6347;
            border: 2px solid #FF6347;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
        }
        input[type="text"], input[type="number"], select {
            background-color: white !important;
            color: black !important;
            border: 2px solid #FF6347;
            border-radius: 5px;
            padding: 12px;
            width: 100%;
            max-width: 250px;
            box-sizing: border-box;
            transition: border-color 0.3s, box-shadow 0.3s;
        }
        input[type="text"]:focus, input[type="number"]:focus, select:focus {
            border-color: #FF4500;
            box-shadow: 0 0 5px rgba(255, 99, 71, 0.5);
            outline: none;
        }
        label {
            color: white;
            font-weight: bold;
            margin-bottom: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# input columns
col1, col2, col3 = st.columns(3)

with col1:
    diagnosis = st.selectbox('Diagnosis (M=1, B=0)', [0, 1], index=0)
    radius_mean = st.number_input('Radius Mean', min_value=0.0, value=17.99, step=0.01)
    texture_mean = st.number_input('Texture Mean', min_value=0.0, value=10.38, step=0.01)
    perimeter_mean = st.number_input('Perimeter Mean', min_value=0.0, value=122.8, step=0.1)

with col2:
    area_mean = st.number_input('Area Mean', min_value=0.0, value=1001.0, step=1.0)
    smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, value=0.1184, step=0.0001)

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
            smoothness_mean, concave_points_mean, symmetry_mean,
            fractal_dimension_mean
        ]]
        Breast_Cancer_prediction = Breast_Cancer_project.predict(features)
    except ValueError as e:
        st.error(f"Prediction error: {str(e)}")

    if Breast_Cancer_prediction[0] == 1:
        Breast_Cancer_diagnosis = 'The person has Malignant Breast Cancer 😷'
    else:
        Breast_Cancer_diagnosis = 'The person has Benign Breast Cancer 😊'

st.success(Breast_Cancer_diagnosis)
