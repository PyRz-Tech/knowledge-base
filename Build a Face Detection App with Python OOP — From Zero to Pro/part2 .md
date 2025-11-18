# Part 2: ConfigModel — Centralized Model Management

## Why We Need ConfigModel
YOLO models are heavy.  
Reloading them inside multiple detectors would:

- Waste CPU/GPU  
- Slow down FPS  
- Increase RAM consumption  
- Cause inconsistent model versions  

So we implement **ConfigModel**, a class that loads models only once and stores them in a dictionary.

---

## How ConfigModel Works

### ✔ `self.models = {}`  
Dictionary that keeps all loaded models.

### ✔ `load_model(name, path)`
Loads a model only if it hasn't been loaded before.

### ✔ `get_model(name)`
Retrieves the loaded model for use in detectors.

---

## Example Code (Your Structure Cleaned & Preserved)
```python
class ConfigModel:
    def __init__(self):
        self.models = {}

    def load_model(self, model_name: str, model_path: str):
        if model_name not in self.models:
            from ultralytics import YOLO
            self.models[model_name] = YOLO(model_path)
        return self.models[model_name]

    def get_model(self, model_name: str):
        return self.models.get(model_name)
```

---

## Why This Architecture Is Good

### ✔ Prevents duplicate loading  
YOLO loads once → all detectors reuse it.

### ✔ Scalable  
You can add as many models as you want:
- YOLOv9-face  
- YOLOv8-pose  
- Segmentation models  
- Custom detectors  

### ✔ Centralized  
All models live in one place → debugging becomes simple.

---

## Conclusion  
ConfigModel is the foundation that keeps the entire system efficient, clean, and scalable.

