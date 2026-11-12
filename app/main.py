import os
import threading

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, Response, UploadFile

from app.detector import Detector, draw

MODEL_PATH = os.getenv("MODEL_PATH", "models/yolo26s.onnx")
THRESHOLDS_PATH = os.getenv("THRESHOLDS_PATH", "models/class_thresholds.json")
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "10")) * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="Road Damage Detection API", version="0.2.0")
detector = Detector(MODEL_PATH, THRESHOLDS_PATH)
inference_lock = threading.Lock()


def load_upload(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail=f"Unsupported type: {file.content_type}")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large")
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="File is not a readable image")
    return image


def run_detection(image, conf, per_class):
    with inference_lock:
        return detector.predict(image, conf=conf, per_class=per_class)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": os.path.basename(MODEL_PATH),
        "classes": list(detector.names.values()),
        "per_class_thresholds": bool(detector.class_conf),
    }


@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=0.99),
    per_class: bool = False,
):
    image = load_upload(file)
    detections, ms = run_detection(image, conf, per_class)
    height, width = image.shape[:2]
    return {
        "image": {"width": width, "height": height},
        "threshold_mode": "per_class" if per_class else f"global {conf}",
        "count": len(detections),
        "inference_ms": round(ms, 1),
        "detections": detections,
    }


@app.post("/predict/annotated")
def predict_annotated(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.01, le=0.99),
    per_class: bool = False,
    show_fps: bool = False,
):
    image = load_upload(file)
    detections, ms = run_detection(image, conf, per_class)
    annotated = draw(image, detections, fps=(1000 / ms) if show_fps else None)
    ok, png = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode the result image")
    return Response(
        content=png.tobytes(),
        media_type="image/png",
        headers={"X-Detections": str(len(detections)), "X-Inference-Ms": f"{ms:.1f}"},
    )