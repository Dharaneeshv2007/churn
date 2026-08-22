# 🚀 Customer Churn Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Customer%20Churn-blueviolet?style=for-the-badge&logo=python" alt="Machine Learning"/>
  <img src="https://img.shields.io/badge/Deep%20Learning-LSTM%20%7C%20GRU-orange?style=for-the-badge&logo=tensorflow" alt="Deep Learning"/>
  <img src="https://img.shields.io/badge/Backend-Flask-black?style=for-the-badge&logo=flask" alt="Flask"/>
  <img src="https://img.shields.io/badge/Frontend-HTML%20%7C%20CSS%20%7C%20JavaScript-yellow?style=for-the-badge&logo=javascript" alt="Frontend"/>
</p>

<p align="center">
  <b>🤖 AI-powered customer churn prediction with explainable insights, risk analysis, CLV estimation, and personalized recommendations.</b>
</p>

<p align="center">

  <a href="https://churn-git-main-dharaneesh-s-projects4.vercel.app">
    <img src="https://img.shields.io/badge/🌐%20Live%20Demo-Vercel-black?style=for-the-badge" alt="Live Demo"/>
  </a>

  <a href="https://churn-dki6.onrender.com">
    <img src="https://img.shields.io/badge/⚡%20API-Render-46E3B7?style=for-the-badge" alt="Backend API"/>
  </a>

  <a href="https://github.com/Dharaneeshv2007/churn">
    <img src="https://img.shields.io/badge/💻%20Source-GitHub-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>

</p>

---

## ✨ Overview

**Customer Churn Prediction System** is a full-stack machine learning application that predicts whether a customer is likely to leave a service.

The system combines **Deep Learning, Explainable AI, Customer Lifetime Value analysis, and recommendation logic** to transform a basic churn prediction into an actionable customer-retention platform.

Instead of simply returning:

> ❌ Customer may churn

the system provides:

* 🎯 Churn probability
* 🚦 Risk level
* ⏳ Estimated time to churn
* 💰 Customer Lifetime Value
* 🔍 SHAP-based explanations
* 💡 Retention recommendations
* 📊 Customer insights

---

## 🎯 Key Features

<table>
<tr>
<td width="50%">

### 🤖 AI Prediction

* Customer churn probability
* LSTM-based prediction
* GRU-based prediction
* Automatic best-model selection
* Model persistence

</td>

<td width="50%">

### 🚦 Risk Analysis

* 🟢 Low Risk
* 🟡 Medium Risk
* 🔴 High Risk
* Estimated time-to-churn
* Risk-based recommendations

</td>
</tr>

<tr>
<td>

### 🔍 Explainable AI

* SHAP explanations
* Top churn reasons
* Feature-level insights
* Customer-specific explanations

</td>

<td>

### 💰 Customer Analytics

* Customer Lifetime Value
* Monthly charge analysis
* Tenure analysis
* Total charges analysis

</td>
</tr>

<tr>
<td>

### 💡 Recommendation Engine

* Risk-based retention actions
* Personalized recommendations
* Actionable customer insights

</td>

<td>

### 🌐 Full-Stack Deployment

* Frontend → Vercel
* Backend → Render
* REST API
* CORS support
* Production-ready architecture

</td>
</tr>
</table>

---

# 🧠 Machine Learning Pipeline

```text
                👤 Customer Data
                       │
                       ▼
              ┌─────────────────┐
              │ Data Validation  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Preprocessing   │
              │                 │
              │ • Encoding      │
              │ • Scaling       │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Deep Learning   │
              │                 │
              │ ┌─────────────┐ │
              │ │    LSTM     │ │
              │ └─────────────┘ │
              │        OR       │
              │ ┌─────────────┐ │
              │ │     GRU     │ │
              │ └─────────────┘ │
              └────────┬────────┘
                       │
                       ▼
              🎯 Churn Probability
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
      🚦 Risk       🔍 SHAP       💰 CLV
      Level       Explanation    Analysis
          │            │            │
          └────────────┼────────────┘
                       ▼
              💡 Recommendation
                       │
                       ▼
                 👤 Customer
```

---

# 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────┐
│                        USER                              │
│                         👤                               │
└─────────────────────────┬────────────────────────────────┘
                          │
                          │ HTTPS
                          ▼
┌──────────────────────────────────────────────────────────┐
│                    VERCEL FRONTEND                       │
│                         🌐                               │
│                                                          │
│             HTML + CSS + JavaScript                     │
└─────────────────────────┬────────────────────────────────┘
                          │
                          │ POST /predict
                          ▼
┌──────────────────────────────────────────────────────────┐
│                    RENDER BACKEND                        │
│                         ⚡                               │
│                                                          │
│                    Flask REST API                        │
│                          │                               │
│        ┌─────────────────┼─────────────────┐             │
│        ▼                 ▼                 ▼             │
│   Preprocessing       ML Model          SHAP             │
│        │                 │                 │             │
│        └─────────────────┼─────────────────┘             │
│                          ▼                               │
│                    Prediction                            │
│                          │                               │
│             ┌────────────┼────────────┐                  │
│             ▼            ▼            ▼                  │
│           Risk          CLV      Recommendation          │
└─────────────────────────┬────────────────────────────────┘
                          │
                          ▼
                  📊 Prediction Result
```

---

# 🛠️ Tech Stack

## 👨‍💻 Programming

<p>
<img src="https://skillicons.dev/icons?i=python,html,css,js" />
</p>

## 🤖 Machine Learning & AI

| Technology      | Purpose                       |
| --------------- | ----------------------------- |
| 🐍 Python       | Core programming              |
| 🐼 Pandas       | Data processing               |
| 🔢 NumPy        | Numerical computation         |
| 📊 Scikit-learn | Preprocessing                 |
| 🧠 TensorFlow   | Deep Learning                 |
| 🔥 Keras        | Neural network implementation |
| 🔍 SHAP         | Explainable AI                |
| 💾 Joblib       | Model/scaler persistence      |

## 🌐 Backend

<p>
<img src="https://skillicons.dev/icons?i=flask,python" />
</p>

* Flask
* Flask-CORS
* Gunicorn
* REST API

## 🎨 Frontend

<p>
<img src="https://skillicons.dev/icons?i=html,css,js" />
</p>

* HTML5
* CSS3
* JavaScript
* Fetch API
* Responsive UI

## ☁️ Deployment

<p>
<img src="https://skillicons.dev/icons?i=vercel,github" />
</p>

* **Vercel** → Frontend
* **Render** → Backend
* **GitHub** → Source control

---

# 📂 Project Structure

```text
churn/
│
├── 📁 backend/
│   │
│   ├── 📄 app.py
│   ├── 📄 requirements.txt
│   │
│   ├── 📁 data/
│   │   └── churnprediction.csv
│   │
│   ├── 📁 model/
│   │   ├── lstm_model.py
│   │   └── gru_model.py
│   │
│   ├── 📁 utils/
│   │   ├── preprocessing.py
│   │   ├── shap_explainer.py
│   │   ├── recommendation.py
│   │   └── clv.py
│   │
│   └── 📁 saved_models/
│       ├── best_model.h5
│       ├── scaler.pkl
│       └── encoder.pkl
│
├── 📁 frontend/
│   ├── 📄 index.html
│   ├── 📄 style.css
│   └── 📄 app.js
│
└── 📄 README.md
```

---

# 📊 Input Features

The model accepts customer information such as:

| Feature             | Description                   |
| ------------------- | ----------------------------- |
| 👤 Gender           | Customer gender               |
| 👴 SeniorCitizen    | Senior citizen indicator      |
| 💑 Partner          | Partner status                |
| 👨‍👩‍👧 Dependents | Dependents status             |
| 📅 Tenure           | Number of months with service |
| 🌐 InternetService  | Internet service type         |
| 📄 Contract         | Contract type                 |
| 💳 MonthlyCharges   | Monthly service charges       |
| 💰 TotalCharges     | Total customer charges        |

---

# 🎯 Prediction Output

The `/predict` endpoint returns information similar to:

```json
{
  "churn_probability": 0.82,
  "risk_level": "High",
  "time_to_churn": "15-30 days",
  "customer_value": 2450.50,
  "recommendation": "Offer a personalized retention plan",
  "top_reasons": [],
  "recommended_action": "Offer a personalized retention plan",
  "prediction_explanation": {}
}
```

---

# 🔌 API Endpoints

## 🏠 Health Check

```http
GET /
```

Returns:

```text
🚀 Customer Churn Backend is Running Successfully
```

---

## ❤️ Backend Health

```http
GET /health
```

Used to verify that the backend is running correctly.

---

## 🎯 Predict Churn

```http
POST /predict
```

### Request

```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "InternetService": "Fiber optic",
  "Contract": "Month-to-month",
  "MonthlyCharges": 75.50,
  "TotalCharges": 906.00
}
```

### Response

```json
{
  "churn_probability": 0.72,
  "risk_level": "Medium",
  "time_to_churn": "30-90 days",
  "customer_value": 1500,
  "recommendation": "...",
  "top_reasons": []
}
```

---

## 🔍 Explain Prediction

```http
POST /explain
```

Generates an explainable prediction using SHAP.

---

## 🧠 Train Model

```http
GET /train
```

The training pipeline:

```text
Dataset
   ↓
Preprocessing
   ↓
LSTM Training
   ↓
GRU Training
   ↓
Compare F1 Scores
   ↓
Select Best Model
   ↓
Save Model
```

---

# 🔍 Explainable AI

A major feature of this project is **Explainable AI**.

Instead of treating the model as a black box, SHAP is used to understand which features contribute to a customer's churn prediction.

```text
Customer
   │
   ▼
Prediction
   │
   ▼
SHAP Analysis
   │
   ├── 📅 Tenure
   ├── 💳 Monthly Charges
   ├── 📄 Contract
   ├── 🌐 Internet Service
   └── 💰 Total Charges
   │
   ▼
Top Churn Reasons
```

This helps businesses understand **why** a customer is likely to churn.

---

# 💰 Customer Lifetime Value

The application also calculates **Customer Lifetime Value (CLV)**.

CLV helps estimate the financial value of retaining a customer.

```text
Customer Data
     │
     ├── Tenure
     │
     └── Monthly Charges
            │
            ▼
       💰 CLV Calculation
            │
            ▼
      Customer Value
```

This allows businesses to prioritize high-value customers who are at high risk of churn.

---

# 🚦 Risk Classification

|   Probability | Risk      | Estimated Time |
| ------------: | --------- | -------------- |
|      `< 0.40` | 🟢 Low    | 90+ days       |
| `0.40 – 0.74` | 🟡 Medium | 30–90 days     |
|      `≥ 0.75` | 🔴 High   | 15–30 days     |

---

# 🤖 Model Selection

The project trains two Deep Learning architectures:

### 🧠 LSTM

Long Short-Term Memory network designed to capture sequential relationships in customer data.

### ⚡ GRU

Gated Recurrent Unit architecture providing a lighter alternative to LSTM.

The system compares their **F1 scores** and automatically selects the better-performing model.

```text
              Dataset
                 │
        ┌────────┴────────┐
        ▼                 ▼
      LSTM               GRU
        │                 │
        ▼                 ▼
     F1 Score          F1 Score
        │                 │
        └────────┬────────┘
                 ▼
          🏆 Best Model
                 │
                 ▼
          best_model.h5
```

---

# ⚙️ Local Installation

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Dharaneeshv2007/churn.git
cd churn
```

## 2️⃣ Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Run Backend

```bash
gunicorn app:app
```

For local development:

```bash
python app.py
```

The backend will run on:

```text
http://127.0.0.1:5000
```

---

# 🌐 Frontend Configuration

The frontend sends prediction requests to:

```javascript
const response = await fetch(
  "https://churn-dki6.onrender.com/predict",
  {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(data)
  }
);
```

For local development, change the backend URL to:

```text
http://127.0.0.1:5000/predict
```

---

# ☁️ Deployment

## Frontend → Vercel

```text
GitHub
   │
   ▼
Vercel
   │
   ▼
🌐 Frontend
```

## Backend → Render

```text
GitHub
   │
   ▼
Render
   │
   ▼
⚡ Flask + Gunicorn
```

### Render Start Command

```bash
gunicorn app:app
```

### Backend URL

<a href="https://churn-dki6.onrender.com">
https://churn-dki6.onrender.com
</a>

---

# 🔐 CORS

The backend is configured to support cross-origin requests from the deployed frontend.

```python
CORS(
    app,
    resources={
        r"/*": {
            "origins": "*"
        }
    }
)
```

This allows the Vercel frontend to communicate with the Render Flask API.

---

# 🧪 Testing

Test the backend health endpoint:

```bash
curl https://churn-dki6.onrender.com/health
```

Test the home endpoint:

```bash
curl https://churn-dki6.onrender.com/
```

For prediction testing, send a JSON `POST` request to:

```text
https://churn-dki6.onrender.com/predict
```

---

# 📸 Screenshots

> Add your project screenshots here.

### 🏠 Home / Prediction Interface

```text
📷 Add screenshot here
```

### 📊 Prediction Result

```text
📷 Add screenshot here
```

### 🔍 Explainable AI

```text
📷 Add screenshot here
```

### 📈 Dashboard / Analytics

```text
📷 Add screenshot here
```

# 💡 Why This Project?

Traditional churn prediction systems often stop at predicting whether a customer will leave.

This project goes further:

```text
Prediction
    +
Risk Analysis
    +
Explainability
    +
Customer Value
    +
Recommendation
    =
🎯 Actionable Churn Intelligence
```

The goal is not only to **predict churn**, but to help businesses understand:

> **Who might churn, why they might churn, how valuable they are, and what action should be taken.**

---

# 👨‍💻 Developer

## Dharaneesh V

🎓 B.Tech Student
🤖 Machine Learning & Deep Learning Enthusiast
💻 Full-Stack Developer

<p align="left">

<a href="https://github.com/Dharaneeshv2007">
  <img src="https://img.shields.io/badge/GitHub-Dharaneeshv2007-181717?style=for-the-badge&logo=github" alt="GitHub"/>
</a>

<a href="https://leetcode.com/u/v_dharaneesh/">
  <img src="https://img.shields.io/badge/LeetCode-Profile-orange?style=for-the-badge&logo=leetcode" alt="LeetCode"/>
</a>

<a href="https://www.hackerrank.com/profile/dharaneeshv7305">
  <img src="https://img.shields.io/badge/HackerRank-Profile-2EC866?style=for-the-badge&logo=hackerrank" alt="HackerRank"/>
</a>

</p>

---

# ⭐ Support

If you found this project useful:

<p align="center">

### ⭐ Star this repository

### 🍴 Fork it

### 💬 Share your feedback

</p>

---

<p align="center">
  <b>Built with ❤️ using Python, TensorFlow, Flask and JavaScript</b>
</p>

<p align="center">
  🚀 <b>Predict • Explain • Act</b> 🚀
</p>
