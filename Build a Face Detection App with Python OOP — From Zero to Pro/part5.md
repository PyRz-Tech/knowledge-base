# Part 5: View Rendering & Main Application (FaceApp)

## Overview
This part covers:

- The **View** class (responsible only for drawing)  
- The **FaceApp** class (main loop + webcam + detection pipeline)  

---

# 🟡 View Class — Everything About Rendering

## Responsibilities
- Draw face ellipse  
- Draw eye and lip landmarks  
- Apply overlays  
- Show result using cv2.imshow  

---

## View Implementation (Your Logic — Cleaned, Structured)
```python
class View:
    def draw_faces(self, img, faces):
        for (x, y, w, h) in faces:
            cx, cy = x + w//2, y + h//2
            cv2.ellipse(img, (cx, cy), (w//2, h//2), 0, 0, 360, (255, 255, 0), 2)
        return img

    def draw_features(self, img, features):
        if not features:
            return img

        for key in features:
            pts = features[key]
            for (x, y) in pts:
                px = int(x * img.shape[1])
                py = int(y * img.shape[0])
                cv2.circle(img, (px, py), 1, (0, 255, 0), -1)

        return img

    def show(self, img):
        cv2.imshow("FaceApp", img)
```

---

# ⚫ FaceApp — The Heart of the System

## Responsibilities

- Load YOLO model (via ConfigModel)  
- Initialize detectors  
- Open webcam (`VideoCapture(0)`)  
- Run real‑time loop  
- Send frames to detectors  
- Render through View  
- Handle quitting (press q)  
- Cleanup webcam + windows  

---

## Full FaceApp Implementation
```python
class FaceApp:
    def __init__(self):
        import cv2
        from app.config_model import ConfigModel
        from app.face_detector import FaceDetector
        from app.face_features_detector import FaceFeaturesDetector
        from app.view import View

        self.cv2 = cv2
        self.view = View()

        cfg = ConfigModel()
        self.model = cfg.load_model("yolo-face", "model/yolov9t-face-lindevs.pt")

        self.face_detector = FaceDetector(self.model)
        self.features_detector = FaceFeaturesDetector()

        self.cap = cv2.VideoCapture(0)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            faces = self.face_detector.detect(frame)
            features = self.features_detector.extract(frame)

            frame = self.view.draw_faces(frame, faces)
            frame = self.view.draw_features(frame, features)
            self.view.show(frame)

            if self.cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        self.cv2.destroyAllWindows()
```

---

## Main Program
```python
if __name__ == "__main__":
    app = FaceApp()
    app.run()
```

---
