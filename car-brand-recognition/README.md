# 🚗 Car Brand Recognition using Convolutional Neural Network (CNN)

This project is a **Deep Learning-based Computer Vision model** that classifies car images into their respective brands.  
It uses a **Convolutional Neural Network (CNN)** and works with datasets containing only:

car_brands_dataset/
train/
test/


Keras automatically generates a validation set (20%) from the training data.

---

# 📂 Dataset Structure

Your dataset must be structured as follows:

car_brands_dataset/
train/
Audi/
BMW/
Mercedes/
Toyota/
test/
Audi/
BMW/
Mercedes/
Toyota/


Each folder contains real car images from that brand.

---

# 🧠 Project Features

- Deep Learning CNN model built using **TensorFlow & Keras**
- Automatic train/validation split (80/20)
- Works with real Kaggle car image datasets
- Displays:
  - Training Accuracy
  - Validation Accuracy
  - Loss graphs
  - Confusion Matrix
  - Classification Report
- Predicts brand for any **single image**
- Predicts and **shows 10 random test images** with confidence scores

---

# 💡 Technologies Used

| Library/Tool | Purpose |
|--------------|---------|
| Python | Programming language |
| TensorFlow / Keras | CNN model |
| OpenCV | Image loading & processing |
| Matplotlib | Plot graphs |
| Seaborn | Heatmaps |
| NumPy | Array processing |
| Scikit-learn | Evaluation metrics |

---

# 🚀 How to Run the Project

## **1. Install required dependencies**


## **2. Place your dataset**


## **3. Run the Jupyter Notebook**

The notebook will:
- Load images  
- Train CNN  
- Validate model  
- Test predictions  
- Show 10 random images with predictions  

---

# 🧩 CNN Model Architecture

The CNN consists of:

- 3 Convolution Layers (32, 64, 128 filters)
- 3 MaxPooling Layers
- Flatten Layer
- Dense Layer (128 units + ReLU)
- Dropout Layer (0.5)
- Output Layer (Softmax for multi-class classification)

Optimizer → **Adam (lr = 0.0001)**  
Loss → **Categorical Crossentropy**

---

# 📊 Example Outputs

## ✔ Accuracy & Loss Plots  
Shows model performance over epochs.

## ✔ Confusion Matrix  
Displays correct vs incorrect predictions.

## ✔ Classification Report  
Precision, Recall, F1-score for each car brand.

## ✔ Single Image Prediction  
Predict brand + confidence.

## ✔ Test 10 Random Images  
Automatic visual testing of model performance.

---

# 🧪 Testing Multiple Images

Use this function to test **10 random images**:

```python
test_and_show_images("car_brands_dataset/test", num_images=10)
