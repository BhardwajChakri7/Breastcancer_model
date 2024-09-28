import pickle
import streamlit as st
from streamlit_option_menu import option_menu


# loading the saved models

breast_cancer = pickle.load(open('BreastCancer_model.sav', 'rb'))
# page title
st.title('Breast Cancer Prediction using ML')


# getting the input data from the user
col1, col2, col3,col4, col5 = st.columns(5)

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

# code for Prediction
breast_cancer_diagnosis = ''

# creating a button for Prediction

if st.button('Breast Cancer Test Button'):
    breast_cancer_prediction = breast_cancer.predict([[Diagnosis,Radius_mean,Texture_mean,Perimeter_mean,Area_mean,Smoothness_mean,Compactness_mean,Contactivity_mean,Concave_points_mean]])
    
    if (breast_cancer_prediction[0] == 1):
      breast_cancer_diagnosis = 'The Breast cancer is Malignant'
    else:
      breast_cancer_diagnosis = 'The Breast Cancer is Benign'
    
st.success(breast_cancer_diagnosis)
