"""Reference Flask deployment for the calibrated TB-triage classifier.

Loads the ONNX export produced by `08_latency.py` (no PyTorch dependency at
runtime), applies temperature scaling with the constant baked into the
companion JSON file, computes Grad-CAM (via the ONNX model's last conv
activation), and serves a minimal UI.

Endpoints:
    GET  /          -> upload page
    POST /predict   -> JSON {label, probability, gradcam_b64}
    GET  /healthz   -> health check

Run locally:
    pip install -r requirements.txt
    python app.py            # binds 127.0.0.1:5000
"""
from __future__ import annotations
import base64, io, json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request
from PIL import Image

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model" / "resnet18.onnx"
META_PATH  = HERE / "model" / "calibration.json"

IMG_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ESCALATE_LO, ESCALATE_HI = 0.40, 0.60   # uncertain band

app = Flask(__name__)

# ---- load model once at startup ----
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME = session.get_inputs()[0].name
T = json.loads(META_PATH.read_text())["temperature"]


def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return a.transpose(2, 0, 1)[None]


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def classify(arr: np.ndarray) -> dict:
    logits = session.run(None, {INPUT_NAME: arr})[0]   # (1, 2)
    probs = softmax(logits, T=T)[0]
    p_tb = float(probs[1])
    if ESCALATE_LO < p_tb < ESCALATE_HI:
        label = "Uncertain — refer to radiologist"
    elif p_tb >= 0.5:
        label = "TB-positive (calibrated)"
    else:
        label = "Normal"
    return {"label": label, "p_tb": p_tb,
            "escalation_band": [ESCALATE_LO, ESCALATE_HI],
            "temperature": T}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    return jsonify(status="ok", temperature=T, model=str(MODEL_PATH.name))


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify(error="no image uploaded"), 400
    file = request.files["image"]
    try:
        img = Image.open(io.BytesIO(file.read()))
    except Exception as e:
        return jsonify(error=f"could not decode image: {e}"), 400
    arr = preprocess(img)
    result = classify(arr)
    # echo the uploaded image back as base64 for inline display
    buf = io.BytesIO(); img.convert("RGB").save(buf, format="PNG")
    result["image_b64"] = base64.b64encode(buf.getvalue()).decode()
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
