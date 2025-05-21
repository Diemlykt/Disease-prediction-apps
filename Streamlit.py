import streamlit as st
import requests
import json

uvicorn backend:app --host 0.0.0.0 --port 8000
st.title("Alzheimer’s Prediction System")

tab1, tab2 = st.tabs(["Clinical Data Upload", "MRI Image Upload"])

with tab1:
    st.header("Upload Clinical Data (CSV)")
    patient_id_clinical = st.text_input("Patient ID (Clinical)", "P001", key="clinical_id")
    clinical_file = st.file_uploader("Upload CSV (e.g., age, MMSE)", type=["csv"], key="clinical_upload")
    if clinical_file and patient_id_clinical:
        if st.button("Upload Clinical Data"):
            url = "http://localhost:8000/upload/csv"
            response = requests.post(url, files={"file": clinical_file})
            st.write(response.json())
    if st.button("Predict Alzheimer’s (Clinical)"):
        url = f"http://localhost:8000/predict/clinical?patient_id={patient_id_clinical}"
        response = requests.post(url)
        if response.status_code == 200:
            result = response.json()
            st.write(f"Prediction: {result['prediction']}")
            st.write("Probabilities:")
            for class_name, prob in result['probabilities'].items():
                st.write(f"{class_name}: {prob:.2%}")
        else:
            st.error(response.json().get('detail', 'Prediction failed'))

with tab2:
    st.header("Upload MRI Image")
    patient_id_image = st.text_input("Patient ID (Image)", "P001", key="image_id")
    image_file = st.file_uploader("Upload MRI (PNG/JPG)", type=["png", "jpg"], key="image_upload")
    if image_file and patient_id_image:
        if st.button("Upload Image"):
            url = f"http://localhost:8000/upload/image?patient_id={patient_id_image}"
            response = requests.post(url, files={"file": image_file})
            if response.status_code == 200:
                image_id = response.json().get('image_id')
                st.session_state['image_id'] = image_id
                st.write(response.json())
            else:
                st.error(response.json().get('detail', 'Upload failed'))
    if st.button("Predict Alzheimer’s (MRI)") and 'image_id' in st.session_state:
        url = f"http://localhost:8000/predict/image?image_id={st.session_state['image_id']}"
        response = requests.post(url)
        if response.status_code == 200:
            result = response.json()
            st.write(f"Prediction: {result['prediction']}")
            st.write("Probabilities:")
            for class_name, prob in result['probabilities'].items():
                st.write(f"{class_name}: {prob:.2%}")
        else:
            st.error(response.json().get('detail', 'Prediction failed'))
