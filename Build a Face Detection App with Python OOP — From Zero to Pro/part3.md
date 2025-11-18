# Part 3: OpenCVBase — Designing a Clean Parent Class

## Why Create a Base Class?
Both FaceDetector and FaceFeaturesDetector must:

- Receive an image (`self.img`)  
- Read images from webcam or file  
- Validate input  
- Provide helper utilities  

Instead of duplicating code, we put common logic in **OpenCVBase**.

---

## Core Responsibilities of OpenCVBase

### ✔ Input management  
Keeps `self.img` consistent across all detectors.

### ✔ Camera handling  
Implements `read_img()` which reads from:
- webcam  
- image path  

### ✔ Error handling  
Ensures detectors never receive invalid images.

---

## Example Implementation
```python
class OpenCVBase:
    def __init__(self):
        self.img = None

    def read_img(self, source=0):
        import cv2

        cap = cv2.VideoCapture(source)
        success, frame = cap.read()
        cap.release()

        if success:
            self.img = frame
            return frame
        return None
```

---

## Why `0` is Default Again?
Because the default webcam is usually device index **0**.

This ensures that if the user runs the app without arguments, it "just works."

---

## Benefits of Using OpenCVBase

### ✔ DRY (Don’t Repeat Yourself)
Only one place to manage images.

### ✔ Consistent API
All detectors expect the same `.img`.

### ✔ Easy to extend
If you add preprocessing later (resize, normalization), you only add it here.

---
