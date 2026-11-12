import json
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

PALETTE = [(255, 99, 71), (65, 105, 225), (60, 179, 113),
           (255, 165, 0), (186, 85, 211), (0, 191, 255)]


class Detector:
    def __init__(self, model_path, thresholds_path=None, imgsz=640):
        self.model = YOLO(str(model_path), task="detect")
        self.imgsz = imgsz
        self.names = self.model.names
        self.class_conf = {}
        if thresholds_path and Path(thresholds_path).exists():
            self.class_conf = json.loads(Path(thresholds_path).read_text())

    def predict(self, image_bgr, conf=0.25, per_class=False):
        use_class = per_class and bool(self.class_conf)
        floor = min(self.class_conf.values()) if use_class else conf

        t0 = time.perf_counter()
        result = self.model.predict(image_bgr, conf=floor, imgsz=self.imgsz,
                                    device="cpu", verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000

        detections = []
        boxes = result.boxes
        for xyxy, cls_id, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
            cls_id = int(cls_id)
            name = self.names[cls_id]
            threshold = self.class_conf.get(name, conf) if use_class else conf
            if score >= threshold:
                detections.append({
                    "class_id": cls_id,
                    "class": name,
                    "confidence": round(score, 3),
                    "box": [round(v, 1) for v in xyxy],
                })
        return detections, ms


def draw(image_bgr, detections, fps=None):
    out = image_bgr.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        color = PALETTE[d["class_id"] % len(PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f'{d["class"]} {d["confidence"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if fps is not None:
        cv2.putText(out, f"{fps:.1f} FPS", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out


if __name__ == "__main__":
    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    assert image is not None, f"Could not read image: {image_path}"

    det = Detector("models/yolo26s.onnx", "models/class_thresholds.json")
    det.predict(image)                      # warm-up (first call is slow)

    for per_class in (False, True):
        dets, ms = det.predict(image, conf=0.25, per_class=per_class)
        print(f"per_class={per_class}: {len(dets)} detections in {ms:.0f} ms")
        for d in dets:
            print("  ", d["class"], d["confidence"], d["box"])

    dets, ms = det.predict(image, per_class=True)
    cv2.imwrite("samples/result.jpg", draw(image, dets, fps=1000 / ms))
    print("saved samples/result.jpg")