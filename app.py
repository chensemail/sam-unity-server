import base64
import io
import requests
import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import traceback

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
    try:
        data = request.json
        if not data or "image" not in data:
            return jsonify({"error": "No image data provided"}), 400

        img_b64 = data["image"]
        
        # 1. Gradio/HF usually needs the Data URI prefix
        if not img_b64.startswith("data:image"):
            img_b64 = f"data:image/png;base64,{img_b64}"

        print("Sending request to HuggingFace...")
        response = requests.post(
            HF_SPACE_URL,
            json={"data": [img_b64]},
            timeout=60 # SAM2 can be slow, don't let it timeout
        )

        # Check if the HF Space is actually awake
        if response.status_code != 200:
            print(f"HF Error: {response.status_code} - {response.text}")
            return jsonify({"error": "HuggingFace Space error", "details": response.text}), 500

        result = response.json()

        # 2. Check if the response has the data we expect
        if "data" not in result or not result["data"]:
            print(f"Unexpected HF Response Format: {result}")
            return jsonify({"error": "HF returned no data"}), 500

        mask_data = result["data"][0]

        # 3. If HF returns a Data URI, strip the header before decoding
        if isinstance(mask_data, str) and "," in mask_data:
            mask_data = mask_data.split(",")[1]

        mask_bytes = base64.b64decode(mask_data)
        mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print("Failed to decode mask image")
            return jsonify({"error": "Invalid mask format from HF"}), 500

        processed = apply_contours(mask)
        mask_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

        print("Success! Sending mask back to Unity.")
        return jsonify({"mask": encode_image(mask_color)})

    except Exception as e:
        # This will print the EXACT line number and error in Render logs
        print("CRITICAL SERVER ERROR:")
        traceback.print_exc() 
        return jsonify({"error": str(e)}), 500


@app.route("/")
def home():
    return "Server running"


if __name__ == "__main__":
    print("Starting Flask server...")
    app.run(host="0.0.0.0", port=10000)