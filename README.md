# Alzheimer’s Disease Prediction Application

## Project Overview

This project is an AI-powered application designed to predict Alzheimer’s disease presence and stage by integrating clinical data and medical imaging analysis. The system assists healthcare professionals by providing reliable, early diagnostic support and disease staging, enhancing patient care and treatment decisions.

The app features two core predictive functions:  
1. **Clinical Data Prediction:** Traditional machine learning models analyze patient clinical information to predict Alzheimer’s disease presence (Yes/No).  
2. **Image-based Prediction:** Deep learning models analyze brain MRI or CT scans to classify the stage of Alzheimer’s disease.

The solution is built using a microservice architecture with a FastAPI backend, Streamlit frontend, and MongoDB for data storage. It is containerized and deployed on Render cloud for scalability and accessibility.

---

## Features

- **User Data Input:** Secure interface for clinical data entry by doctors or users.  
- **Image Upload:** Upload brain MRI/CT images for Alzheimer’s stage prediction.  
- **Predictive Models:**  
  - Machine Learning (Random Forest, SVM) for clinical data prediction.  
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
| Containerization| Docker                   |

---

## Machine Learning Models

### Clinical Data Prediction

- Algorithms: Logistic Regression, Support Vector Machine (SVM), Random Forest.  
- Input Features: Demographic and clinical variables related to Alzheimer’s.  
- Model Selection: Best model chosen using hyperparameter tuning and cross-validation.  
- Evaluation Metrics: Precision, Recall, F1-score, Classification reports.

### Image-based Prediction

- Model: Custom CNN implemented in PyTorch for multi-class Alzheimer’s stage classification.  
- Preprocessing: Image resizing, normalization, augmentation.  
- Evaluation: Accuracy, confusion matrix, and explainability visualizations.

---

## Application Architecture

- **Microservice Design:** Backend API serves model inferences and manages database. Frontend interacts via RESTful API calls.  
- **Data Flow:**  
  1. User inputs clinical data or uploads brain images via frontend.  
  2. Frontend sends data to FastAPI backend API.  
  3. Backend performs model inference.  
  4. Prediction results are sent back to frontend and stored in MongoDB.  
  5. Users can view and optionally download prediction reports.

- **Deployment:** Docker containers run the backend and frontend, hosted on Render cloud platform for continuous availability.

---

## Usage Instructions

1. Access the app frontend via the deployed URL.  
2. For clinical prediction: fill the patient data form and submit.  
3. For image prediction: upload a brain MRI or CT scan and submit.  
4. View the prediction results instantly on the interface.  
5. Optionally download a detailed report for record-keeping.  
6. All input and prediction data is securely saved in MongoDB.

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


## Future Work

- Add user authentication and role-based access control.  
- Expand dataset to improve image model accuracy and robustness.  
- Implement downloadable PDF reports with visualizations and explanations.  
- Develop notification system for prediction updates and reminders.  
- Extend support for other neurodegenerative diseases and comorbid conditions.

---

Thank you for reviewing this Alzheimer’s Disease Prediction project!

---

