# 🧠 Alzheimer’s Disease Prediction App

A full-stack medical application that assists healthcare professionals in predicting Alzheimer’s disease using either structured clinical data or MRI images. The app provides interactive prediction tools, result storage, and report generation for enhanced diagnosis support.

---

## 🚀 Features

- **Clinical Data Prediction**: Upload structured clinical information (CSV) to predict Alzheimer's diagnosis.
- **Image-Based Stage Classification**: Upload MRI images to predict the stage of Alzheimer’s disease (e.g., Non-Demented, Very Mild, Mild, Moderate).
- **Dual Upload Modes**:
  - Clinical data upload via CSV file.
  - Medical image upload via PNG/JPG.
- **MongoDB Integration**: Stores both clinical records and MRI images (via GridFS).
- **Prediction Result Display**: Presents prediction outputs clearly within the web UI.
- **FastAPI RESTful API**: A scalable backend to process predictions.
- **Streamlit Frontend**: A user-friendly interface to interact with prediction modules.
- **PDF Report Generation** *(optional feature)*: Export results for documentation or patient records.
- **Render Deployment**: Full deployment with uptime maintenance using CRON job-based pings.

---

## 🧠 Machine Learning Models

### 1. Clinical Data-Based Prediction

- **Algorithm**: Logistic Regression or similar classifier trained on structured features.
- **Input Features** (sample):
  - Gender
  - Age
  - Education
  - SES (Socioeconomic Status)
  - MMSE (Mini-Mental State Exam)
  - eTIV (Estimated Total Intracranial Volume)
  - nWBV (Normalized Whole Brain Volume)
  - ASF (Atlas Scaling Factor)

### 2. MRI Image-Based Stage Classification

- **Algorithm**: Convolutional Neural Network (CNN)
- **Input**: MRI Brain Scan (PNG/JPG)
- **Output Classes**:
  - Non-Demented
  - Very Mild Demented
  - Mild Demented
  - Moderate Demented

---

## ⚙️ Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Database**: MongoDB (with GridFS for image storage)
- **Machine Learning**: Scikit-learn, TensorFlow/Keras
- **Deployment**: Render (with CRON job for uptime)
- **Others**:
  - `pandas`, `numpy` for data handling
  - `Pillow` for image processing
  - `requests` for API communication

---

## 📁 File Structure

