import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load the saved model
breast_cancer = pickle.load(open('Breast_Cancer_model.sav', 'rb'))

# Use the raw GitHub link for the background image
page_bg_img = '''
<style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://github.com/SHAIK-RAIYAN-2022-CSE/malaria/blob/main/Images-free-abstract-minimalist-wallpaper-HD.jpg?raw=true");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0); /* Transparent header */
    }
    input {
        background-color: white !important;
        color: black !important;
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 10px;
        font-size: 16px;
        width: 100%;
        margin-bottom: 10px;
    }
    h1 {
        color: white;
        text-align: center;
    }
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Page title
st.markdown("<h1 style='text-align: center;'>Breast Cancer Prediction using Machine Learning</h1>", unsafe_allow_html=True)

# Input section with border and blur
with st.container():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)
