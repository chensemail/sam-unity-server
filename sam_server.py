import os
import base64
import io
import torch
import gdown
import numpy as np
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import gradio as gr

# ---------- Download checkpoint from Google Drive if not exists ----------
CHECKPOINT_URL = "https://drive.google.com/uc?id=1ThYP9Hc3TVFJSURias2Ju9932-WAjOw8"
CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"

if not os.path.exists(CHECKPOINT_PATH):
    print("Downloading SAM checkpoint...")
    gdown.download(CHECKPOINT_URL, CHECKPOINT_PATH, quiet=False)

# ---------- Initialize SAM ----------
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=CHECKPOINT_PATH)
sam.to("cuda" if torch.cuda.is_available() else "cpu")
mask_generator = SamAutomaticMaskGenerator(sam)

# ---------- Helper functions ----------
def decode_image(img_bytes):
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
        if cv2.contourArea(cntr) > 300:
            convHull = cv2.convexHull(cntr)
            cv2.drawContours(blackImg, [convHull], -1, 255, 1)
    return blackImg

def segment_image(file):
    img_bytes = file.read()
    img = decode_image(img_bytes)

    # 1️⃣ Run SAM
    masks = mask_generator.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    bw_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for m in masks:
        bw_mask[m["segmentation"]] = 255

    # 2️⃣ Apply CV2 processing
    processed_mask = apply_erosion_and_contours(bw_mask)

    # 3️⃣ Convert to BGR
    mask_color = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)
    return mask_color  # Gradio will return as image

# ---------- Gradio Interface ----------
demo = gr.Interface(
    fn=segment_image,
    inputs=gr.File(label="Input Image"),
    outputs=gr.Image(label="Mask"),
    live=False
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
