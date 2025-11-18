# Part 4: Detection Pipeline — YOLO & FaceMesh

## Overview
This part explains the full logic behind:

- **FaceDetector** (YOLOv9t‑face)  
- **FaceFeaturesDetector** (MediaPipe FaceMesh)  

Both detectors extend **OpenCVBase** and work together.

---

# 🔴 FaceDetector — YOLOv9 Face Detection

## Responsibilities
- Run YOLO on input frame  
- Extract bounding boxes  
- Convert YOLO output format  
- Provide clean face coordinates  

---

## YOLO Prediction Flow
YOLO returns detections in the following format:

```
[x1, y1, x2, y2, confidence, class_id]
```

We convert these into:

```
(x, y, w, h)
```

---

## Cleaned Version of Your Detector Logic
```python
class FaceDetector(OpenCVBase):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def detect(self, img):
        self.img = img
        results = self.model(img)[0]

        faces = []
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            w, h = x2 - x1, y2 - y1
            faces.append((x1, y1, w, h))

        return faces
```

---

# 🟣 FaceFeaturesDetector — MediaPipe FaceMesh

## Responsibilities

- Extract all 539 landmarks  
- Select subsets belonging to:
  - left eye  
  - right eye  
  - outer lips  
  - inner lips  
- Return a structured dictionary

---

## Why `refine_landmarks=True`?
Because it improves accuracy on:

- eyelids  
- iris  
- lips  

It specifically sharpens fine-grained areas — perfect for your project.

---

## Landmark Index Groups (Preserving Your Data)

### 👁 Left Eye  
Indices: **33, 7, 163, 144, 145, 153, 154, 155, 133**

### 👁 Right Eye  
Indices: **263, 249, 390, 373, 374, 380, 381, 382, 362**

### 👄 Outer Lips  
Indices: **61–88**

### 👄 Inner Lips  
Indices: **0–18**

*(These match your custom extracted subsets.)*

---

## Example Detector Code
```python
class FaceFeaturesDetector(OpenCVBase):
    def __init__(self):
        super().__init__()
        import mediapipe as mp
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True
        )

    def extract(self, img):
        self.img = img
        results = self.mesh.process(img)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark

        def pts(idxs):
            return [(landmarks[i].x, landmarks[i].y) for i in idxs]

        features = {
            "left_eye": pts([33,7,163,144,145,153,154,155,133]),
            "right_eye": pts([263,249,390,373,374,380,381,382,362]),
            "outer_lip": pts(range(61,89)),
            "inner_lip": pts(range(0,19))
        }
        return features
```

---

