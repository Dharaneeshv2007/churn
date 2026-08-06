# 🚀 Customer Churn Prediction using Deep Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-DeepLearning-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask)
![HTML5](https://img.shields.io/badge/HTML5-Frontend-red?style=for-the-badge&logo=html5)
![CSS3](https://img.shields.io/badge/CSS3-Styling-blue?style=for-the-badge&logo=css3)
![JavaScript](https://img.shields.io/badge/JavaScript-Interactive-yellow?style=for-the-badge&logo=javascript)

### 🔍 AI-Powered Customer Churn Prediction Dashboard

Predict customer churn risk using **LSTM & GRU Deep Learning Models**, explain predictions with **SHAP Explainable AI**, and generate intelligent customer retention recommendations.

</div>

---

## 📖 Project Overview

Customer churn is one of the major challenges faced by telecom and service-based companies. Losing existing customers can significantly impact business revenue and growth.

This project uses **Deep Learning techniques** to analyze customer information and predict whether a customer is likely to leave the company. The system also provides churn probability, risk level, customer value estimation, explainable insights, and retention recommendations.

---

## 🎯 Objectives

✅ Predict customer churn accurately using deep learning

✅ Compare LSTM and GRU models

✅ Identify high-risk customers

✅ Generate explainable predictions using SHAP

✅ Provide retention recommendations

✅ Improve customer satisfaction and reduce revenue loss

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| TensorFlow | Deep Learning Framework |
| Keras | Model Building |
| Flask | Backend API |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| Scikit-Learn | Data Preprocessing |
| SHAP | Explainable AI |
| HTML | Frontend Structure |
| CSS | UI Design |
| JavaScript | Frontend Logic |

---

## 📂 Dataset Features

The model uses the following customer attributes:

- Gender
- Senior Citizen
- Partner
- Dependents
- Tenure
- Internet Service
- Contract Type
- Monthly Charges
- Total Charges

### 🎯 Target Variable

**Churn**

- Yes → Customer leaves the company
- No → Customer stays with the company

---

## 🏗️ Project Architecture

```text
Customer Input Data
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Scaling
        │
        ▼
LSTM / GRU Model
        │
        ▼
Churn Prediction
        │
        ▼
SHAP Explainable AI
        │
        ▼
Recommendation System
        │
        ▼
Dashboard Output
```

---

## ⚙️ Modules

### 📥 Customer Input Module
Collects customer details from the dashboard.

### 🧹 Data Preprocessing Module
Cleans, transforms, and encodes customer data.

### 📊 Feature Scaling Module
Normalizes numerical values for improved learning.

### 🧠 LSTM / GRU Module
Predicts customer churn probability.

### 🔍 Explainable AI Module
Identifies key factors influencing churn predictions.

### 💡 Recommendation Module
Suggests customer retention strategies.

### 📈 Dashboard Module
Displays prediction results and insights.

---

## 🧠 Deep Learning Models

### LSTM (Long Short-Term Memory)

- Learns long-term customer behavior patterns
- Handles complex relationships in customer data
- Improves prediction accuracy

### GRU (Gated Recurrent Unit)

- Faster training compared to LSTM
- Requires fewer parameters
- Efficient for churn prediction tasks

### Best Model Selection

The model with the highest **F1-Score** is selected automatically.

---

## 📊 Performance Metrics

The project evaluates performance using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Sample Results

| Metric | Value |
|----------|----------|
| Accuracy | 73.86% |
| Precision | 50.55% |
| Recall | 75.54% |
| F1 Score | 60.57% |

---

## 🔍 Explainable AI (SHAP)

The system uses SHAP values to explain:

- Why a customer may churn
- Important influencing features
- Model decision transparency

Example:

```text
Top Reasons:
✔ Contract Type
✔ Monthly Charges
✔ Tenure
```

---

## 🎯 Risk Levels

| Probability | Risk Level |
|-------------|------------|
| 0 – 40% | Low |
| 40 – 75% | Medium |
| Above 75% | High |

---

## 💡 Retention Recommendations

| Risk Level | Recommendation |
|------------|---------------|
| Low | No Action Needed |
| Medium | Customer Engagement |
| High | Offer Discounts & Support |

---

## 🚀 How to Run the Project

### Clone Repository

```bash
git clone https://github.com/dharaneeshvijayakumar2007-png/HCL.git
```

### Navigate to Project Folder

```bash
cd HCL
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Backend

```bash
python app.py
```

### Open Frontend

Open:

```text
index.html
```

in your browser.

---

## 📷 Project Screenshots

### Dashboard

(Add Screenshot Here)

### Prediction Output

(Add Screenshot Here)

### Explainable AI Output

(Add Screenshot Here)

---

## 📈 Future Enhancements

- Real-time churn monitoring
- Email alert system
- Cloud deployment
- Customer segmentation
- Advanced recommendation engine
- Multi-industry churn prediction

---

## 🎓 Academic Use

This project was developed as part of an academic mini/major project for learning:

- Deep Learning
- Explainable AI
- Customer Analytics
- Business Intelligence
- Predictive Modeling

---

## 👨‍💻 Developer

**Dharaneesh V**

### Connect with Me

[![GitHub](https://img.shields.io/badge/GitHub-Visit_Profile-black?style=for-the-badge&logo=github)](https://github.com/dharaneeshvijayakumar2007-png)

---

## ⭐ Support

If you found this project useful:

⭐ Star this repository

🍴 Fork this repository

📢 Share with others

---

<div align="center">

### 🚀 Predict Customer Churn Before It Happens

Made with ❤️ using TensorFlow, Keras, Flask & Explainable AI

</div>

---

## Backend Deployment Guide

This folder now contains the complete Flask backend for production deployment on Render.

### Installation

```bash
cd backend
pip install -r requirements.txt
```

### Run Locally

```bash
cd backend
python app.py
```

### Render Deployment

Use this start command:

```bash
gunicorn app:app
```

The backend expects its assets relative to the backend folder:

- Dataset: `data/churnprediction.csv`
- Model: `saved_models/best_model.h5`
- Scaler: `saved_models/scaler.pkl`
- Encoder: `saved_models/encoder.pkl`

### API Endpoints

- `POST /predict` - returns churn probability, risk level, CLV, explanations, and recommendation
- `POST /train` - retrains the LSTM/GRU models and saves the best model artifacts
