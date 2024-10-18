import pickle
import streamlit as st

# Load the saved model
breast_cancer = pickle.load(open('Breast_Cancer_model.sav', 'rb'))

# CSS styles
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
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Page title
st.markdown('<h1>Breast Cancer Prediction using Machine Learning</h1>', unsafe_allow_html=True)

with st.container():
    col1, col2 = st.columns(2)

    with col1:
        Diagnosis = st.text_input('Diagnosis', placeholder='Enter 0 or 1')
        Radius_mean = st.text_input('Radius Mean', placeholder='e.g., 14.5')
        Texture_mean = st.text_input('Texture Mean', placeholder='e.g., 19.0')
        Perimeter_mean = st.text_input('Perimeter Mean', placeholder='e.g., 85.0')

    with col2:
        Area_mean = st.text_input('Area Mean', placeholder='e.g., 500.0')
        Smoothness_mean = st.text_input('Smoothness Mean', placeholder='e.g., 0.1')
        Compactness_mean = st.text_input('Compactness Mean', placeholder='e.g., 0.5')
        Contactivity_mean = st.text_input('Contactivity Mean', placeholder='e.g., 0.7')
        Concave_points_mean = st.text_input('Concave Points Mean', placeholder='e.g., 0.3')

# Prediction
breast_cancer_diagnosis = ''

# Prediction button
if st.button('🔍 Predict', help='Click to predict breast cancer'):
    features = [
        float(Diagnosis), float(Radius_mean), float(Texture_mean), 
        float(Perimeter_mean), float(Area_mean), float(Smoothness_mean), 
        float(Compactness_mean), float(Contactivity_mean), float(Concave_points_mean)
    ]
    breast_cancer_prediction = breast_cancer.predict([features])
    
    if breast_cancer_prediction[0] == 1:
        breast_cancer_diagnosis = 'The Breast Cancer is Malignant 😷'
    else:
        breast_cancer_diagnosis = 'The Breast Cancer is Benign 😊'

st.success(breast_cancer_diagnosis)
