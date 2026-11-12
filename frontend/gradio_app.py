import os
import time

import cv2
import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

PALETTE = [(255, 99, 71), (65, 105, 225), (60, 179, 113),
           (255, 165, 0), (186, 85, 211), (0, 191, 255)]


def draw(image_rgb, detections):
    out = image_rgb.copy()
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d["box"])
        color = PALETTE[d["class_id"] % len(PALETTE)]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f'{d["class"]} {d["confidence"]:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def detect(image, conf, per_class):
    if image is None:
        raise gr.Error("Please upload or capture an image first.")

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr)
    if not ok:
        raise gr.Error("Could not encode the image.")

    t0 = time.perf_counter()
    try:
        response = requests.post(
            f"{API_URL}/predict",
            params={"conf": conf, "per_class": str(per_class).lower()},
            files={"file": ("image.jpg", buffer.tobytes(), "image/jpeg")},
            timeout=60,
        )
    except requests.exceptions.RequestException:
        raise gr.Error(f"Cannot reach the API at {API_URL}. Is the FastAPI server running?")
    round_trip_ms = (time.perf_counter() - t0) * 1000

    if response.status_code != 200:
        raise gr.Error(f"API error {response.status_code}: {response.text}")

    result = response.json()
    detections = result["detections"]
    annotated = draw(image, detections)

    summary = (
        f"Detections: {result['count']}  |  "
        f"Model time: {result['inference_ms']} ms  |  "
        f"Request round-trip: {round_trip_ms:.0f} ms "
        f"(~{1000 / round_trip_ms:.1f} FPS end-to-end)  |  "
        f"Threshold: {result['threshold_mode']}"
    )
    table = [[d["class"], d["confidence"], d["box"]] for d in detections]
    return annotated, summary, table


with gr.Blocks(title="Road Damage Detection") as demo:
    gr.Markdown("# Road Damage Detection\nUpload or capture a road photo. "
                "The image is sent to the FastAPI backend (YOLO26s).")
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(label="Input image", type="numpy", sources=["upload", "webcam"])
            conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05, label="Confidence threshold")
            per_class = gr.Checkbox(value=False,
                                    label="Use tuned per-class thresholds (ignores the slider)")
            run = gr.Button("Detect", variant="primary")
        with gr.Column():
            image_out = gr.Image(label="Result")
            info = gr.Textbox(label="Summary")
            table = gr.Dataframe(headers=["class", "confidence", "box [x1, y1, x2, y2]"],
                                 label="Detections")

    run.click(detect, inputs=[image_in, conf, per_class], outputs=[image_out, info, table])

if __name__ == "__main__":
    demo.launch()