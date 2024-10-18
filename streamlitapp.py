import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load the saved model
breast_cancer = pickle.load(open('Breast_Cancer_model.sav', 'rb'))

# Page title with styling
st.markdown(
    """
    <style>
        .title {
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            color: white;
        }
        .content {
            border: 1px solid rgba(255, 255, 255, 0.5);
            border-radius: 10px;
            backdrop-filter: blur(10px);
            padding: 20px;
            margin: 20px auto;
            width: 80%;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Breast Cancer Prediction using Machine Learning</div>', unsafe_allow_html=True)

# Container for input data
with st.container():
    st.markdown('<div class="content">', unsafe_allow_html=True)

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
        breast_cancer_prediction = breast_cancer.predict([[Diagnosis, Radius_mean, Texture_mean, Perimeter_mean, Area_mean, Smoothness_mean, Compactness_mean, Contactivity_mean, Concave_points_mean]])
        
        if breast_cancer_prediction[0] == 1:
            breast_cancer_diagnosis = 'The Breast Cancer is Malignant 😷'
        else:
            breast_cancer_diagnosis = 'The Breast Cancer is Benign 😊'

    st.success(breast_cancer_diagnosis)

    st.markdown('</div>', unsafe_allow_html=True)
