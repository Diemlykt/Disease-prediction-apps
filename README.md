# Alzheimer’s Disease Prediction Application

## Project Overview

This project is an AI-powered application designed to predict Alzheimer’s disease presence and stage by integrating clinical data and medical imaging analysis. The primary focus is on building a functional, end-to-end system, including a user-friendly interface, backend API, and cloud deployment, rather than training optimized machine learning models. Due to a 3-day time constraint, pre-trained models from Kaggle were used to expedite development, ensuring a working prototype that assists healthcare professionals with early diagnostic support and disease staging.

The app features two core predictive functions:  
1. **Clinical Data Prediction:** Traditional machine learning models analyze patient clinical information to predict Alzheimer’s disease presence (Yes/No).  
2. **Image-based Prediction:** Deep learning models analyze brain MRI scans to classify the stage of Alzheimer’s disease

The solution is built using a microservice architecture with a FastAPI backend, Streamlit frontend, and MongoDB for data storage. It is containerized and deployed on Render cloud for scalability and accessibility.

---

## Features

- **User Data Input:** Secure interface for clinical data entry by doctors or users.  
- **Image Upload:** Upload brain MRI images for Alzheimer’s stage prediction.  
- **Predictive Models:**  
  - Machine Learning (XGBoost) for clinical data prediction.  
  - Deep Learning CNN for Alzheimer’s stage classification from images.  
- **Database Integration:** MongoDB stores all input data and prediction results.  
- **Backend API:** FastAPI handles inference requests and database operations.  
- **Frontend Interface:** Streamlit provides a user-friendly UI for inputs and predictions.  
- **Deployment:** Dockerized app hosted on Render for reliable service availability.

---

## Technology Stack

| Component       | Technology/Library       |
|-----------------|-------------------------|
| Backend         | FastAPI (Python)         |
| Frontend        | Streamlit (Python)       |
| Database        | MongoDB (NoSQL)          |
| Machine Learning| scikit-learn, PyTorch    |
| Deployment      | Render                   |

---

## Machine Learning Models

### Clinical Data Prediction

- Algorithms: "Decision Tree", "Random Forest", "K-Nearest Neighbors", "Logistic Regression", "Support Vector Machine", "XGBoost"
- Input Features: Demographic and clinical variables related to Alzheimer’s.  
- Model Selection: Best model chosen using hyperparameter tuning and cross-validation.  
- Evaluation Metrics: Precision, Recall, F1-score, Classification reports.

### Image-based Prediction

- Model: Pre-trained CNN sourced from Kaggle (https://www.kaggle.com/code/metinusta/oasis-alzheimer-s-detection-training), implemented in PyTorch for multi-class Alzheimer’s stage classification.  
- Preprocessing: Image resizing, normalization.  
- Evaluation: Accuracy.

---

## Application Architecture

- **Microservice Design:** Backend API serves model inferences and manages database. Frontend interacts via RESTful API calls.  
- **Data Flow:**  
  1. User inputs clinical data or uploads brain images via frontend.  
  2. Frontend sends data to FastAPI backend API.  
  3. Backend performs model inference.  
  4. Prediction results are sent back to frontend and stored in MongoDB.  
  5. Users can view and optionally download prediction reports.

- **Deployment:** The application is deployed on Render using direct Python script deployment with a requirements.txt file. Docker was not used due to time constraints, which may limit scalability but simplifies setup for the prototype. A health check script (ping.py) ensures reliability.

---

## Usage Instructions

1. Access the app frontend via the deployed URL.  
2. For clinical prediction: fill the patient data form and submit.  
3. For image prediction: upload a brain MRI or CT scan and submit.  
4. View the prediction results instantly on the interface.  
5. Optionally download a detailed report for record-keeping.  
6. All input and prediction data is securely saved in MongoDB.

---

## User Interface
The Streamlit frontend provides an intuitive interface for healthcare professionals to input clinical data or upload brain MRI/CT scans. Prediction results are displayed instantly, with options to view confidence scores and download reports. The interface is designed for ease of use, ensuring seamless integration into clinical workflows.
Scan the QR code below to access the deployed Alzheimer’s Disease Prediction Application. Use file in example_file to test the app
### Application QR Code
<img src="https://raw.githubusercontent.com/Diemlykt/Disease-prediction-apps/792f0e08711ed0fcd7621c38c2f4688386a1bfd7/Alzheimer_qrcode.png" alt="QR Code" width="150" height="150">

Figure 1: Screenshot of the Streamlit interface showing the clinical data input form and prediction output. 

### Clinical data upload
![CSV upload](https://github.com/Diemlykt/Disease-prediction-apps/blob/179610fd4cb15403a889fd889b82771eeb74c08d/Screenshot/Screenshot%20Clinical-1.jpg)

Figure 2: Screenshot of the Streamlit interface showing the image prediction. 
### Image upload
![Image upload](https://github.com/Diemlykt/Disease-prediction-apps/blob/179610fd4cb15403a889fd889b82771eeb74c08d/Screenshot/Screenshot%20Image%20update.jpg)

---

## Folder Structure
```
AlzheimerProject/
├── Model/                   # Contains models and training utilities
│ ├── CNN.pt                 # Trained CNN model for image-based prediction
│ ├── dataset.py             # Dataset loader and transformer for image data
│ ├── model2use.py           # Model loading and prediction functions
│ ├── scaler.pkl             # Scaler used for normalizing clinical data
│ ├── train_test_val_running.py # Script for training and evaluating models
│ ├── xgboost_model.json     # Trained XGBoost model (clinical data, serialized in JSON)
│ └── xgboost_model.pkl      # Trained XGBoost model (pickle format)
├── .gitattributes           # Git attributes for consistent repository behavior
├── AlzheimerProject.ipynb   # Jupyter Notebook for exploratory analysis and prototyping
├── README.md                # Project documentation and instructions
├── Streamlit.py             # Streamlit frontend application script
├── backend.py               # FastAPI backend script to handle requests
├── ping.py                  # Script to verify the backend is live and responsive (health check)
├── requirements.txt         # Python dependencies list
├── Screenshot               # screenshot of apps
├── example_file             # CSV files and images that you can upload and test the app
```
---
### Description of Key Files

- **CNN.pt** – Pretrained deep learning model for image-based Alzheimer’s stage prediction.
- **dataset.py** – Contains dataset class and transformations for model training and inference.
- **model2use.py** – Handles loading and applying trained models (both CNN and XGBoost).
- **scaler.pkl** – Preprocessing scaler used on numerical clinical data before model input.
- **train_test_val_running.py** – Training pipeline for traditional and deep learning models.
- **xgboost_model.json / .pkl** – Trained XGBoost models for predicting Alzheimer’s from clinical data.
- **Streamlit.py** – UI for users to input data and receive prediction results.
- **backend.py** – FastAPI app that processes incoming data and routes to appropriate models.
- **ping.py** – Script to verify the backend is live and responsive (health check).
- **AlzheimerProject.ipynb** – Jupyter notebook used during model development and experimentation.

---
## Limitation
Due to a 3-day time constraint, the project faced the following limitations:
- Data Acquisition: Limited time prevented custom data scraping or collection. The project relies on publicly available Kaggle datasets, which ensured quick access but may introduce biases. Healthcare datasets often require lengthy approval processes, which were not feasible within the timeline.
- Model Development: The project prioritizes system integration over model training. Pre-trained XGBoost and CNN models from Kaggle were used instead of developing new models, limiting customization for specific use cases and reliance on the quality of Kaggle-sourced models.
- Deployment: Docker was not used due to time constraints and complexity, with the app deployed directly on Render using Python scripts. This may limit scalability and reproducibility compared to containerized deployment.
- Model Robustness: The pre-trained models were not fine-tuned or validated on external datasets due to time constraints, potentially reducing generalizability to real-world clinical scenarios. Future work includes collecting custom datasets, fine-tuning models, and implementing Docker for improved deployment.

---

## Future Work

- Implement user authentication and role-based access control.
- Collect and integrate diverse datasets (e.g., ADNI) for improved model performance.
- Fine-tune pre-trained models for specific use cases.
- Add downloadable PDF reports with visualizations.
- Develop a notification system for prediction updates.
- Extend support for other neurodegenerative diseases.
- Incorporate Docker for scalable deployment.

---

Thank you for reviewing this Alzheimer’s Disease Prediction project!

---

