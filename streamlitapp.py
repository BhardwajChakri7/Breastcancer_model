import pickle
import streamlit as st

# Load the saved model
breast_cancer = pickle.load(open('Breast_Cancer_model.sav', 'rb'))

# CSS Styles
page_bg_img = '''
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
    input[type="number"] {
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
    input[type="number"]:focus {
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
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Page title
st.markdown('<h1>Breast Cancer Prediction using Machine Learning</h1>', unsafe_allow_html=True)

# 3-column layout for inputs
col1, col2, col3 = st.columns(3)

with col1:
    Diagnosis = st.number_input('Diagnosis (0 or 1)', min_value=0, max_value=1, value=0)
    Radius_mean = st.number_input('Radius Mean', min_value=5.0, max_value=30.0, value=14.5)
    Smoothness_mean = st.number_input('Smoothness Mean', min_value=0.0, max_value=1.0, value=0.1)

with col2:
    Texture_mean = st.number_input('Texture Mean', min_value=5.0, max_value=50.0, value=19.0)
    Perimeter_mean = st.number_input('Perimeter Mean', min_value=50.0, max_value=200.0, value=85.0)
    Compactness_mean = st.number_input('Compactness Mean', min_value=0.0, max_value=1.0, value=0.5)

with col3:
    Area_mean = st.number_input('Area Mean', min_value=100.0, max_value=3000.0, value=500.0)
    Contactivity_mean = st.number_input('Contactivity Mean', min_value=0.0, max_value=1.0, value=0.7)
    Concave_points_mean = st.number_input('Concave Points Mean', min_value=0.0, max_value=1.0, value=0.3)

# Prediction
breast_cancer_diagnosis = ''

# Prediction button
if st.button('🔍 Predict', help='Click to predict breast cancer'):
    features = [
        Diagnosis, Radius_mean, Texture_mean, 
        Perimeter_mean, Area_mean, Smoothness_mean, 
        Compactness_mean, Contactivity_mean, Concave_points_mean
    ]
    breast_cancer_prediction = breast_cancer.predict([features])
    
    if breast_cancer_prediction[0] == 1:
        breast_cancer_diagnosis = 'The Breast Cancer is Malignant 😷'
    else:
        breast_cancer_diagnosis = 'The Breast Cancer is Benign 😊'

st.success(breast_cancer_diagnosis)
