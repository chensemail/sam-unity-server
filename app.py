import base64
import io
import requests
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

# HuggingFace Space API
HF_SPACE_URL = "https://skalskip-segment-anything-model-2.hf.space/run/predict"

# -------- Decode base64 image ----------
def decode_image(img_base64):
    img_bytes = base64.b64decode(img_base64)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img_pil)

# -------- Encode image ----------
def encode_image(img_cv):
    _, buffer = cv2.imencode(".png", img_cv)
    return base64.b64encode(buffer).decode("utf-8")

# -------- OpenCV contour processing ----------
def apply_contours(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    eroded = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)

    contours,_ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x,y = mask.shape
    result = np.zeros((x,y),dtype=np.uint8)

    for cntr in contours:
        area = cv2.contourArea(cntr)
        if area > 300:
            hull = cv2.convexHull(cntr)
            cv2.drawContours(result,[hull],-1,255,1)

    return result

# -------- API endpoint ----------
@app.route("/segment", methods=["POST"])
def segment():

    data = request.json
    img_b64 = data["image"]

    img = decode_image(img_b64)

    # Send image to SAM2 HuggingFace space
    response = requests.post(
        HF_SPACE_URL,
        json={
            "data":[img.tolist()]
        }
    )

    sam_mask = np.array(response.json()["data"][0])

    processed = apply_contours(sam_mask)

    mask_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    return jsonify({
        "mask": encode_image(mask_color)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)