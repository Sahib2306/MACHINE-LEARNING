# ❤️ Heart Attack Risk Prediction using Machine Learning

This project applies **Machine Learning classification models** to predict the likelihood of a **heart attack** based on multiple lifestyle, demographic, and medical features.  
The dataset used contains **50,000 rows and 30 features**, including age groups, diet, alcohol consumption, physical activity levels, smoking habits, obesity index, and more.

---

## 📌 Project Overview

This notebook performs:

### ✔ Exploratory Data Analysis (EDA)
- Checking shape, missing values, column types  
- Visualizing distributions  
- Understanding correlations

### ✔ Data Cleaning & Preprocessing
- Handling categorical & numerical features  
- Ordinal encoding  
- One-hot encoding  
- Outlier treatment  
- Train–test split  
- Balancing classes using **SMOTE**

### ✔ Machine Learning Models Used
1. **K-Nearest Neighbors (KNN)**  
2. **Decision Tree Classifier**  
3. **Support Vector Classifier (SVC)** – RBF + Linear  
4. **Random Forest Classifier**  
5. **GridSearchCV** for hyperparameter tuning  

Each model is evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Classification Report  

---

## 📁 Dataset

The dataset used:
heart_attack_russia_youth_vs_adult.csv


It contains **youth and adult** data for heart attack risk factors in Russia.

---

## 🧠 Machine Learning Pipeline

1. Load dataset  
2. EDA & data inspection  
3. Identify ordinal vs nominal categorical data  
4. Apply custom ordinal mapping (Healthy → 3 → etc.)  
5. One-hot encode remaining nominal variables  
6. Train–test split (80/20)  
7. Handle class imbalance with **SMOTE**  
8. Train different ML models  
9. Compare results  
10. Analyze feature importance (Random Forest)

---

## 📊 Model Performance

The project generates:
- Model-wise performance reports  
- Best parameters from GridSearch  
- Feature importance bar charts  

*(You can later add tables or screenshots if uploading to GitHub.)*

---

## 🚀 Technologies Used

- **Python**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **Statsmodels**
- **Imbalanced-Learn (SMOTE)**

---

## 📦 How to Run


OR install manually:
pip install pandas numpy seaborn matplotlib scikit-learn statsmodels imbalanced-learn

Then open the notebook:
jupyter notebook cardio_risk_prediction.ipynb
