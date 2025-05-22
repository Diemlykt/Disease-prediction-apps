import streamlit as st
import requests
import pandas as pd
import json

# FastAPI URL (update for production or ngrok for Colab)
FASTAPI_URL = "https://disease-prediction-apps.onrender.com"  

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
            response = requests.post(f"{FASTAPI_URL}/predict/clinical", params={"patient_id": int(st.session_state['clinical_patient_id'][0])})
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
    image_file = st.file_uploader("Upload MRI (PNG/JPG)", type=["png", "jpg", "jpeg"], key="image_upload")
    
    if image_file and patient_id_image:
        if st.button("Upload Image"):
            with st.spinner("Uploading image..."):
                try:
                    # Prepare the request
                    files = {
                        "file": (image_file.name, image_file.getvalue(), image_file.type)
                    }
                    data = {
                        "patient_id": patient_id_image  # Send as form data
                    }

                    # Make the request
                    response = requests.post(
                        f"{FASTAPI_URL}/upload/image",
                        files=files,
                        data=data  # Changed from params to data
                    )
                    
                    # Handle response
                    try:
                        result = response.json()
                    except ValueError:
                        st.error("Server returned invalid response format")
                        st.code(response.text, language='text')
                        

                    if response.status_code == 200:
                        if result.get('success', False):
                            st.success(result.get('message', result.get('status', 'Image uploaded successfully')))
                            st.session_state['uploaded_image_id'] = result['image_id']
                            st.session_state['image_patient_id'] = patient_id_image
                        else:
                            st.error(result.get('detail', 'Upload failed (server reported failure)'))
                    else:
                        st.error(f"Upload failed (HTTP {response.status_code}): {result.get('detail', 'Unknown error')}")

                except requests.exceptions.RequestException as e:
                    st.error(f"Network error: {str(e)}")
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
                    st.error(traceback.format_exc())
    
    # if st.button("Predict Alzheimer’s (MRI)") and 'uploaded_image_id':
    #     try:
    #         # Request image prediction
    #         response = requests.post(f"{FASTAPI_URL}/predict/image", params={"image_id": st.session_state['uploaded_image_id']})
    #         if response.status_code == 200:
    #             result = response.json()
    #             st.write(f"**Prediction**: {result['prediction']}")
    #             st.write("**Probabilities**:")
    #             for class_name, prob in result['probabilities'].items():
    #                 st.write(f"{class_name}: {prob:.2%}")
    #         else:
    #             st.error(f"Prediction failed: {response.json().get('detail', 'Unknown error')}")
    #     except Exception as e:
    #         st.error(f"Error: {str(e)}")


    if st.button("Predict Alzheimer's (MRI)"):
        # Validate session state
        if 'uploaded_image_id' not in st.session_state:
            st.warning("Please upload an MRI image first")
            st.stop()
    
        with st.spinner("Analyzing MRI scan..."):
            try:
                response = requests.post(
                    f"{FASTAPI_URL}/predict/image",
                    params={"image_id": st.session_state['uploaded_image_id']},
                    timeout=45  # Increased timeout for model inference
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Create two columns for better layout
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.subheader("Diagnosis")
                        # Color code based on prediction
                        status_color = {
                            'Non-Demented': 'green',
                            'Very Mild Demented': 'blue',
                            'Mild Demented': 'orange',
                            'Moderate Demented': 'red'
                        }.get(result['prediction'], 'gray')
                        
                        st.markdown(f"""
                        <div style='
                            border-left: 5px solid {status_color};
                            padding: 1rem;
                            margin: 1rem 0;
                        '>
                            <h3 style='color: {status_color}; margin-top: 0;'>
                                {result['prediction']}
                            </h3>
                            <p>Confidence: {max(result['probabilities'].values()):.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.subheader("Probability Breakdown")
                        for class_name, prob in sorted(
                            result['probabilities'].items(),
                            key=lambda x: x[1],
                            reverse=True
                        ):
                            # Create a progress bar with label
                            st.write(f"**{class_name}**")
                            st.progress(prob, text=f"{prob:.2%}")
                
                else:
                    error_detail = response.json().get('detail', 'Unknown error')
                    st.error(f"Prediction failed: {error_detail}")
                    
            except requests.exceptions.Timeout:
                st.error("Analysis timed out. The model might be processing other requests.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")
