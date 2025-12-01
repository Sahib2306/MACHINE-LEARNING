# 🚗 Car Price Prediction using Deep Learning (Edge-Optimized)

This project builds an **end-to-end Deep Learning model** to predict used car prices and then **optimizes the model for edge deployment** using TensorFlow Lite (TFLite) quantization.  
The project includes data preprocessing, EDA, ML baseline model, Deep Neural Network (DNN), and edge optimization.

---

## 📌 Features

- 🔍 **Data Cleaning & Preprocessing**
- 📊 **Exploratory Data Analysis (EDA)** with heatmaps and scatter plots
- 🌲 **Baseline ML Model**: Random Forest Regressor
- 🤖 **Deep Learning Model** using TensorFlow/Keras (DNN)
- ⚙️ **Model Optimization**  
  - Architecture simplification  
  - Dropout regularization  
  - Early stopping  
  - TensorFlow Lite quantization (Float32 → INT8)
- ⚡ **Model Size & Inference Speed Comparison**
- 📈 **Evaluation Metrics:** MAE, RMSE, R²

---


---

## 🧠 Tech Stack

- **Python**
- **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**
- **Scikit-Learn**
- **TensorFlow / Keras**
- **TensorFlow Lite (TFLite)**

---

## 📊 Dataset

You can use the CarDekho dataset from Kaggle:

https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho

Rename the CSV to:

car_prices.csv




## 🚀 How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/Car-Price-Prediction-DeepLearning-Optimized.git
cd Car-Price-Prediction-DeepLearning-Optimized


pip install -r requirements.txt
jupyter notebook
notebook/Car_Price_Prediction_Optimized.ipynb
