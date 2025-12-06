import os
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
import torch
import gdown
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ----------- Download checkpoint from Google Drive if not exists ----------
CHECKPOINT_URL = "https://drive.google.com/file/d/1ThYP9Hc3TVFJSURias2Ju9932-WAjOw8/view?usp=sharing"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"

if not os.path.exists(CHECKPOINT_PATH):
    print("Downloading SAM checkpoint...")
    gdown.download(CHECKPOINT_URL, CHECKPOINT_PATH, quiet=False)

# ----------- Initialize SAM ----------
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=CHECKPOINT_PATH)
sam.to("cuda" if torch.cuda.is_available() else "cpu")

mask_generator = SamAutomaticMaskGenerator(sam)

# ----------- Flask app ----------
app = Flask(__name__)
CORS(app)

# ---------- Helper functions ----------
def decode_image(img_base64):
    img_bytes = base64.b64decode(img_base64)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img_pil)[:, :, ::-1]  # RGB -> BGR

def encode_image(img_cv):
    _, buffer = cv2.imencode(".png", img_cv)
    return base64.b64encode(buffer).decode("utf-8")

def apply_erosion_and_contours(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)

    contours, _ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y = img.shape
    blackImg = np.zeros((x, y), dtype=np.uint8)
    for cntr in contours:
        area = cv2.contourArea(cntr)
        if area > 300:
            convHull = cv2.convexHull(cntr)
            cv2.drawContours(blackImg, [convHull], -1, 255, 1)
    return blackImg

# ---------- API endpoint ----------
@app.route("/segment_cv2", methods=["POST"])
def segment_cv2():
    data = request.json
    img_b64 = data["image"]
    img = decode_image(img_b64)

    # 1️⃣ Run SAM
    masks = mask_generator.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    bw_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for m in masks:
        bw_mask[m["segmentation"]] = 255

    # 2️⃣ Apply CV2 processing
    processed_mask = apply_erosion_and_contours(bw_mask)

    # 3️⃣ Convert to BGR for Unity
    mask_color = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)
    return jsonify({"mask": encode_image(mask_color)})

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render will set PORT automatically
    app.run(host="0.0.0.0", port=port)
