# Part 1: Introduction & Project Architecture

## Overview
This project implements a complete **face detection + facial feature extraction pipeline** using:

- **YOLOv9-face** for face bounding box detection  
- **MediaPipe FaceMesh** for detailed landmark extraction (eyes, lips, etc.)  
- A clean **Object‑Oriented (OOP) architecture** to keep everything modular, extensible, and production‑ready.

The tutorial explains everything in depth — what each class does, why it exists, and how all pieces work together.

---

## Why OOP for a CV project?
### ✔ Single Responsibility  
Each class handles *one* job:
- Detecting faces  
- Extracting features  
- Rendering  
- Managing models  
- Running the main loop  

### ✔ DRY Principles  
Common functionality (image handling, webcam reading, etc.) lives in a single parent class: **OpenCVBase**.

### ✔ Scalability  
Add a new model? New feature extractor? New rendering theme?  
Just create a new class without touching existing ones.

### ✔ Performance  
ConfigModel loads models **once**, preventing heavy repetitive initialization.

---

## Project Structure
```
project/
│
├── app/
│   ├── config_model.py
│   ├── opencv_base.py
│   ├── face_detector.py
│   ├── face_features_detector.py
│   ├── view.py
│   ├── face_app.py
│
├── model/
│   └── yolov9t-face-lindevs.pt
│
└── main.py
```

---

## Component Responsibilities

### 🔵 ConfigModel
Loads and caches models (YOLO, etc). Prevents loading multiple times.

### 🟢 OpenCVBase  
Parent class that provides:
- `self.img`  
- `read_img()`  
- camera reading logic  
All detectors inherit from it.

### 🔴 FaceDetector  
Uses YOLO to detect faces and returns bounding boxes.

### 🟣 FaceFeaturesDetector  
Uses MediaPipe FaceMesh to extract:
- Left eye landmarks  
- Right eye landmarks  
- Outer lips  
- Inner lips  

### 🟡 View  
Only responsible for drawing:
- Ellipses around faces  
- Feature overlays  
- Weighted blending  
- Rendering with cv2.imshow  

### ⚫ FaceApp  
Main "application controller":
- Opens webcam (`VideoCapture(0)`)  
- Runs detection pipeline  
- Sends results to View  
- Handles exit logic  
- Cleans up resources  

---

## `VideoCapture(0)` — Why Zero?
`0` represents the **default webcam**.

If you have multiple cameras:
- `0` → laptop camera  
- `1` → USB webcam  
- `2` → virtual camera  

This project uses webcam input in both OpenCVBase and FaceApp.

---
