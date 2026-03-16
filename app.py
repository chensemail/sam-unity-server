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

HF_SPACE_URL = "https://skalskip-segment-anything-model-2.hf.space/gradio_api/run/predict"


# -------- Decode base64 image ----------
def decode_image(img_base64):
    print("Decoding base64 image...")
    img_bytes = base64.b64decode(img_base64)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = np.array(img_pil)
    print("Image decoded:", img.shape)
    return img


# -------- Encode image ----------
def encode_image(img_cv):
    print("Encoding result mask...")
    _, buffer = cv2.imencode(".png", img_cv)
    return base64.b64encode(buffer).decode("utf-8")


# -------- OpenCV contour processing ----------
def apply_contours(mask):
    print("Applying OpenCV contour processing...")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21,21))
    eroded = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)

    contours,_ = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    x,y = mask.shape
    result = np.zeros((x,y),dtype=np.uint8)

    count = 0
    for cntr in contours:
        area = cv2.contourArea(cntr)
        if area > 300:
            hull = cv2.convexHull(cntr)
            cv2.drawContours(result,[hull],-1,255,1)
            count += 1

    print("Contours drawn:", count)
    return result


# -------- API endpoint ----------
@app.route("/segment", methods=["POST"])
def segment():

    print("===== New request received =====")

    data = request.json

    if "image" not in data:
        print("ERROR: No image in request")
        return jsonify({"error":"no image"}),400

    img_b64 = data["image"]

    img = decode_image(img_b64)

    print("Sending image to HuggingFace SAM2...")

    response = requests.post(
        HF_SPACE_URL,
        json={"data":[img_b64]}
    )

    print("HuggingFace response status:", response.status_code)

    result = response.json()

    print("HuggingFace response received")

    mask_b64 = result["data"][0]
    print("Decoding mask...")

    mask_bytes = base64.b64decode(mask_b64)
    mask = cv2.imdecode(np.frombuffer(mask_bytes,np.uint8),cv2.IMREAD_GRAYSCALE)

    print("Mask shape:", mask.shape)

    processed = apply_contours(mask)

    mask_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    print("Returning mask to Unity")

    return jsonify({
        "mask": encode_image(mask_color)
    })


@app.route("/")
def home():
    return "Server running"


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=10000)