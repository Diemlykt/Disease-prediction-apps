import streamlit as st
import requests
import pandas as pd
import json

# FastAPI URL (update for production or ngrok for Colab)
FASTAPI_URL = "https://disease-prediction-apps.onrender.com"  # Replace with ngrok URL for Colab testing

st.title("Alzheimer’s Detection System")

# Tabs for clinical and imaging inputs
tab1, tab2 = st.tabs(["Clinical Data Upload", "MRI Image Upload"])

with tab1:
    st.header("Upload Clinical Data (CSV)")
    # patient_id_clinical = st.text_input("Patient ID (Clinical)", "P001", key="clinical_id")
    clinical_file = st.file_uploader("Upload CSV (32 features: Age, MMSE, BMI, etc.)", type=["csv"], key="clinical_upload")
    
    # if clinical_file and patient_id_clinical:
    if clinical_file:
        if st.button("Upload Clinical Data"):
            try:
                # Send CSV to FastAPI
                response = requests.post(f"{FASTAPI_URL}/upload/csv", files={"file": clinical_file})
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"{result['status']}")
                    st.write(f"Record IDs: {', '.join(result['ids'])}")
                    st.session_state['clinical_patient_id'] = result['patient_ids']

                else:
                    st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # if st.button("Predict Alzheimer’s (Clinical)") and 'clinical_patient_id' in st.session_state:
    if st.button("Predict Alzheimer’s (Clinical)"):
        try:
            # Request clinical prediction
            response = requests.post(f"{FASTAPI_URL}/predict/clinical", params={"patient_id": int(st.session_state['clinical_patient_id'])})
            if response.status_code == 200:
                result = response.json()
                st.write(f"**Prediction**: {result['prediction']}")
                st.write("**Probabilities**:")
                for class_name, prob in result['probabilities'].items():
                    st.write(f"{class_name}: {prob:.2%}")
            else:
                st.error(f"Prediction failed: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tab2:
    st.header("Upload MRI Image")
    patient_id_image = st.text_input("Patient ID (Image)", "P001", key="patient_id_image")
    image_file = st.file_uploader("Upload MRI (PNG/JPG)", type=["png", "jpg"], key="image_upload")
    
    if image_file and patient_id_image:
        if st.button("Upload Image"):
            try:
                # Send image to FastAPI
                
                # response = requests.post(
                #     f"{FASTAPI_URL}/upload/image",
                #     params={"patient_id": patient_id_image},
                #     files={"file": image_file}
                # )
                # if response.status_code == 200:
                #     result = response.json()
                #     st.success(f"{result['status']}")
                #     st.session_state['uploaded_image_id'] = result['image_id']
                #     st.session_state['image_patient_id'] = patient_id_image
                # else:
                #     st.error(f"Upload failed: {response.json().get('detail', 'Unknown error')}")

                response = requests.post(f"{FASTAPI_URL}/upload/image",
                    params={"patient_id": patient_id_image},
                    files={"file": image_file}
                )
                
                try:
                    result = response.json()
                except ValueError:
                    st.error("Server did not return a valid JSON response.")
                    st.error(f"Response text: {response.text}")
                    raise
                
                if response.status_code == 200:
                    st.success(f"{result['status']}")
                    st.session_state['uploaded_image_id'] = result['image_id']
                    st.session_state['image_patient_id'] = patient_id_image
                else:
                    st.error(f"Upload failed: {result.get('detail', 'Unknown error')}")

               
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.button("Predict Alzheimer’s (MRI)") and 'uploaded_image_id':
        try:
            # Request image prediction
            response = requests.post(f"{FASTAPI_URL}/predict/image", params={"image_id": st.session_state['uploaded_image_id']})
            if response.status_code == 200:
                result = response.json()
                st.write(f"**Prediction**: {result['prediction']}")
                st.write("**Probabilities**:")
                for class_name, prob in result['probabilities'].items():
                    st.write(f"{class_name}: {prob:.2%}")
            else:
                st.error(f"Prediction failed: {response.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
