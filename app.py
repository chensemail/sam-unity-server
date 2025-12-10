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

# ---------- Download checkpoint if missing ----------
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"  # change to your checkpoint file
CHECKPOINT_URL = "https://drive.google.com/file/d/1NnjLIf-kbcYLZz8-_MKnAshnguZAZH-S/view?usp=sharing"

if not os.path.exists(CHECKPOINT_PATH):
    print("Downloading SAM checkpoint...")
    gdown.download(CHECKPOINT_URL, CHECKPOINT_PATH, quiet=False)

# ---------- Initialize SAM ----------
model_type = "vit_b"  # or "vit_h" if you want
sam = sam_model_registry[model_type](checkpoint=CHECKPOINT_PATH)
sam.to("cuda" if torch.cuda.is_available() else "cpu")
mask_generator = SamAutomaticMaskGenerator(sam)

# ---------- Helper functions ----------
def decode_base64_image(img_b64: str):
    img_bytes = base64.b64decode(img_b64)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img_pil)[:, :, ::-1]  # RGB -> BGR

def encode_image_to_base64(img_cv):
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

# ---------- Gradio function for Unity ----------
def segment_image_base64(img_b64: str):
    img = decode_base64_image(img_b64)

    # Run SAM
    masks = mask_generator.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    bw_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for m in masks:
        bw_mask[m["segmentation"]] = 255

    # Apply CV2 processing
    processed_mask = apply_erosion_and_contours(bw_mask)

    # Convert to BGR
    mask_color = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)
    return encode_image_to_base64(mask_color)

# ---------- Gradio Interface ----------
demo = gr.Interface(
    fn=segment_image_base64,
    inputs=gr.Textbox(label="Base64 Image"),
    outputs=gr.Textbox(label="Mask Base64"),
    live=False
)

# ---------- Launch on Railway ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        enable_queue=True,
        api_mode=True
    )
