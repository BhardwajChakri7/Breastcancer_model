import pickle
import streamlit as st

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
        font-size: 48px; /* Increased font size for the title */
        font-weight: bold;
    }
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Page title with larger font size
st.markdown('<h1>Breast Cancer Prediction using Machine Learning</h1>', unsafe_allow_html=True)

# Container for input data
with st.container():
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        Diagnosis = st.text_input('Diagnosis')
        
    with col2:
        Radius_mean = st.text_input('Radius Mean')

    with col3:
        Texture_mean = st.text_input('Texture Mean')
        
    with col4:
        Perimeter_mean = st.text_input('Perimeter Mean')
        
    with col5:
        Area_mean = st.text_input('Area Mean')

    with col1:
        Smoothness_mean = st.text_input('Smoothness Mean')
        
    with col2:
        Compactness_mean = st.text_input('Compactness Mean')

    with col3:
        Contactivity_mean = st.text_input('Contactivity Mean')
        
    with col4:
        Concave_points_mean = st.text_input('Concave Points Mean')

    # Prediction
    breast_cancer_diagnosis = ''

    # Prediction button
    if st.button('🔍 Breast Cancer Test Button'):
        breast_cancer_prediction = breast_cancer.predict([[
            Diagnosis, Radius_mean, Texture_mean, Perimeter_mean, 
            Area_mean, Smoothness_mean, Compactness_mean, 
            Contactivity_mean, Concave_points_mean
        ]])
        
        if breast_cancer_prediction[0] == 1:
            breast_cancer_diagnosis = 'The Breast Cancer is Malignant 😷'
        else:
            breast_cancer_diagnosis = 'The Breast Cancer is Benign 😊'

    st.success(breast_cancer_diagnosis)
