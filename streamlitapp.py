import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load the saved model
breast_cancer = pickle.load(open('Breast_Cancer_model.sav', 'rb'))

# Use the raw GitHub link for the background image
page_bg_img = '''
<style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://github.com/BhardwajChakri7/Breastcancer_model/blob/main/Images-free-abstract-minimalist-wallpaper-HD.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0); /* Transparent header */
    }
    .block-container {
        max-width: 800px;
        margin: 50px auto;
        padding: 20px;
        border: 2px solid #ccc;
        border-radius: 15px;
        background: rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(10px);
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.6);
    }
    input {
        background-color: white !important;
        color: black !important;
        border-radius: 10px;
        border: 1px solid #ccc;
        padding: 10px;
        font-size: 16px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 24px;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: white;
        color: #4CAF50;
        border: 2px solid #4CAF50;
    }
    h1, h2, h3, h4, h5, h6, p {
        color: white !important;
    }
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Page title
st.markdown("<h1 style='text-align: center;'>Breast Cancer Prediction using Machine Learning</h1>", unsafe_allow_html=True)

# Input section with a subheader
st.subheader("Enter Patient's Clinical Data")

# Organize inputs into rows and columns
with st.container():
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        diagnosis = st.text_input('Diagnosis')
        smoothness_mean = st.text_input('Smoothness Mean')

    with col2:
        radius_mean = st.text_input('Radius Mean')
        compactness_mean = st.text_input('Compactness Mean')

    with col3:
        texture_mean = st.text_input('Texture Mean')
        contactivity_mean = st.text_input('Contactivity Mean')

    with col4:
        perimeter_mean = st.text_input('Perimeter Mean')
        concave_points_mean = st.text_input('Concave Points Mean')

    with col5:
        area_mean = st.text_input('Area Mean')

# Prediction result
breast_cancer_diagnosis = ''

# Prediction button
if st.button('🔍 Predict Breast Cancer'):
    try:
        cancer_prediction = breast_cancer.predict([[diagnosis, radius_mean, texture_mean, perimeter_mean,
                                                    area_mean, smoothness_mean, compactness_mean,
                                                    contactivity_mean, concave_points_mean]])

        if cancer_prediction[0] == 1:
            breast_cancer_diagnosis = 'The Breast Cancer is Malignant 😷'
        else:
            breast_cancer_diagnosis = 'The Breast Cancer is Benign 😊'
    except ValueError as e:
        st.error(f"Prediction error: {str(e)}")

# Display result
st.success(breast_cancer_diagnosis)
