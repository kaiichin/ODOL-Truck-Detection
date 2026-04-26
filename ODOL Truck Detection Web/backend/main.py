from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import numpy as np
import cv2
import base64
from PIL import Image
from io import BytesIO

app = FastAPI()

frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# --- Load AI Models on startup ---
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.layers import Dense

print("Downloading models from Hugging Face...")
best_pt = hf_hub_download("Kaiichin/odol-truck_detector", "best.pt")
v3_h5 = hf_hub_download("Kaiichin/odol-truck_detector", "mobilenet_odol_classifierV3.h5")

yolo_model = YOLO(best_pt)
print("Brain 1 (YOLOv8) loaded!")

class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

classifier_model = tf.keras.models.load_model(v3_h5, custom_objects={'Dense': CustomDense})
print("Brain 2 (MobileNetV2) loaded!")

@app.get("/")
def home():
    return FileResponse(str(frontend_path / "index.html"))

@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Stage 1: YOLO finds the truck
    results = yolo_model(img, conf=0.3, verbose=False)

    if len(results[0].boxes) == 0:
        return {"detected": False}

    # Get best detection
    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])

    # Stage 2: Crop and classify
    crop = img[y1:y2, x1:x2]
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    crop_pil = crop_pil.resize((224, 224))
    crop_array = np.array(crop_pil).astype("float32") / 255.0
    prediction = float(classifier_model.predict(np.expand_dims(crop_array, 0), verbose=0)[0][0])

    label = "ODOL" if prediction > 0.5 else "NORMAL"
    confidence = prediction if prediction > 0.5 else 1 - prediction

    # Draw result on image
    color = (0, 0, 255) if label == "ODOL" else (0, 200, 0)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
    cv2.putText(img, f"{label} ({confidence:.0%})", (x1, y1-15),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    # Convert to base64 for frontend
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "detected": True,
        "label": label,
        "confidence": round(confidence, 4),
        "annotated_image": img_base64
    }
