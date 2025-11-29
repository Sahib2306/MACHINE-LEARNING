# **Smart Traffic Management & Violation Detection 🚦🚨**
*A Real-Time Computer Vision System for Helmet Detection & Signal Violation Analysis*

---

## 📌 **Project Overview**

This project implements a **Smart Traffic Monitoring System** using **Computer Vision + Deep Learning** to automatically detect:

- **Helmet Violations** (motorbike rider without helmet)  
- **Signal Violations** (vehicle crossing stop-line on red light)  
- **Vehicle Detection & Counting** (car, bike, bus, truck)

The system processes a **video feed (CCTV/Webcam)**, detects objects using **YOLOv8**, applies rule-based logic, and saves violation evidence with timestamps.

---

## 🎯 **Key Features**

### ✔ **Real-Time Object Detection**
Detects vehicles, persons, and helmets using YOLOv8.

### ✔ **Helmet Violation Detection**
If a rider is detected on a motorbike **without wearing a helmet**, the system logs a violation.

### ✔ **Signal Violation Detection**
Vehicles crossing a **virtual stop-line** when the traffic signal is red are flagged as violators.

### ✔ **Violation Proof Capture**
The system automatically saves:
- Annotated violation images  
- CSV logs with timestamp, frame number, and violation type  

### ✔ **Syllabus Mapping (Perfect for CA + Final Project)**
- **Unit II:** Frame extraction, resizing, ROI  
- **Unit III:** Deep learning with YOLO  
- **Unit VI:** Integrated application + performance metrics  

---

## 🛠️ **Tech Stack**

| Component | Technology |
|----------|------------|
| **Programming Language** | Python 3 |
| **Computer Vision Library** | OpenCV |
| **ML Model** | YOLOv8 (Ultralytics) |
| **Data Storage** | CSV logs |
| **Execution Environment** | Local system / GPU machine |

---

