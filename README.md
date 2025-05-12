# 🧠 Stroke Risk Predictor
A machine learning model trained to predict stroke events using 11 clinical features such as age, BMI, glucose levels, and lifestyle indicators.

## 📊 Dataset Overview
This model is trained on a publicly available Stroke Prediction dataset containing clinical and demographic features. Each entry represents a patient's health profile, and the model predicts whether or not the patient is at risk of stroke.

### 🔍 Features Used:
| Feature           | Description |
|------------------|-------------|
| id               | Unique identifier |
| gender           | "Male", "Female", or "Other" |
| age              | Age of the patient |
| hypertension     | 0 = No, 1 = Yes |
| heart_disease    | 0 = No, 1 = Yes |
| ever_married     | "No" or "Yes" |
| work_type        | "children", "Govt_job", "Never_worked", "Private", "Self-employed" |
| Residence_type   | "Rural" or "Urban" |
| avg_glucose_level| Average blood glucose level |
| bmi              | Body Mass Index |
| smoking_status   | "formerly smoked", "never smoked", "smokes", or "Unknown" |
| stroke           | 0 = No Stroke, 1 = Stroke (Target variable) |

## 🧠 Model Overview
- Supervised binary classification task
- Algorithms: Logistic Regression / Random Forest / XGBoost (customizable)
- Performance metrics: Accuracy, ROC AUC, F1 Score

> This model can help in early stroke risk assessment using routine clinical data.

## 📁 Files
- `stroke_model.py`: Contains the model pipeline including preprocessing, training, and prediction

## 🚀 Getting Started

### 🔧 Requirements
Install dependencies (create a `requirements.txt` if not present):
`pip install pandas scikit-learn matplotlib seaborn joblib`



## ▶️ Run the Model
To train and test the model on the dataset:
`python stroke_model.py`
Make sure your dataset (stroke_data.csv) is in the same directory or adjust the path in the script.

📈 Future Improvements
- Web-based risk prediction form using Streamlit
- Hyperparameter tuning with GridSearchCV
- Feature importance and SHAP explainability
- Model deployment (Flask, FastAPI, or Streamlit Cloud)

📚 Dataset: Kaggle Stroke Prediction Dataset: https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset
 
📬Created by Amaan Arif – feel free to reach out for questions or collaboration ideas!
