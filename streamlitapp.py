import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load the saved models
Malaria_Project = pickle.load(open('malaria_model1.sav', 'rb'))
breast_cancer = pickle.load(open('Breast_Cancer_model.sav', 'rb'))

# Background image and styling
page_bg_img = '''
<style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://github.com/BhardwajChakri7/Breastcancer_model/blob/main/Images-free-abstract-minimalist-wallpaper-HD.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    [data-testid="stHeader"] {
        background: rgba(0, 0, 0, 0);
    }
    .content-container {
        max-width: 900px;
        margin: 50px auto;
        padding: 30px;
        border: 2px solid rgba(255, 255, 255, 0.5);
        border-radius: 15px;
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(15px);
        box-shadow: 0px 4px 30px rgba(0, 0, 0, 0.7);
    }
    input {
        background-color: white !important;
        color: black !important;
        border-radius: 8px;
        border: 1px solid #ccc;
        padding: 12px;
        font-size: 16px;
        margin-bottom: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 12px 28px;
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
        text-align: center;
    }
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Malaria Prediction Section
st.markdown("<h1>Malaria Prediction using Machine Learning</h1>", unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        Temperature_Above_Avg = st.text_input('Temperature Above Avg')
        Insecticide_Use = st.text_input('Insecticide Use')

    with col2:
        High_Rainfall = st.text_input('High Rainfall')
        Health_Facilities_Adequate = st.text_input('Health Facilities Adequate')

    with col3:
        High_Humidity = st.text_input('High Humidity')
        Vaccination_Rate_High = st.text_input('Vaccination Rate High')

    with col4:
        High_Population_Density = st.text_input('High Population Density')
        Mosquito_Net_Coverage_High = st.text_input('Mosquito Net Coverage High')

    with col5:
        Malaria_Outbreak = st.text_input('Malaria Outbreak')

    Malaria_diagnosis = ''
    if st.button('🔍 Malaria Disease Test'):
        try:
            prediction = Malaria_Project.predict([[
                Temperature_Above_Avg, High_Rainfall, High_Humidity,
                High_Population_Density, Malaria_Outbreak, Insecticide_Use,
                Health_Facilities_Adequate, Vaccination_Rate_High, Mosquito_Net_Coverage_High
            ]])
            if prediction[0] == 1:
                Malaria_diagnosis = 'The person is affected with Malaria 😷'
            else:
                Malaria_diagnosis = 'The person is not affected with Malaria 😊'
        except ValueError as e:
            st.error(f"Prediction error: {str(e)}")

    st.success(Malaria_diagnosis)
    st.markdown('</div>', unsafe_allow_html=True)

# Breast Cancer Prediction Section
st.markdown("<h1>Breast Cancer Prediction using Machine Learning</h1>", unsafe_allow_html=True)
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

    breast_cancer_diagnosis = ''
    if st.button('🔍 Predict Breast Cancer'):
        try:
            cancer_prediction = breast_cancer.predict([[
                diagnosis, radius_mean, texture_mean, perimeter_mean,
                area_mean, smoothness_mean, compactness_mean,
                contactivity_mean, concave_points_mean
            ]])
            if cancer_prediction[0] == 1:
                breast_cancer_diagnosis = 'The Breast Cancer is Malignant 😷'
            else:
                breast_cancer_diagnosis = 'The Breast Cancer is Benign 😊'
        except ValueError as e:
            st.error(f"Prediction error: {str(e)}")

    st.success(breast_cancer_diagnosis)
    st.markdown('</div>', unsafe_allow_html=True)
