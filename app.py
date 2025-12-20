import os
import base64
import io
import torch
import gdown
import numpy as np
import cv2
from PIL import Image
import gradio as gr

# MobileSAM imports
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

# ---------- Download MobileSAM checkpoint from Google Drive ----------
CHECKPOINT_PATH = "mobile_sam_weights_only.pt"
GDRIVE_FILE_ID = "1SzJMrgsHU_wQfjqE3iA_ikdVS4Fkg6V1"
GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"

if not os.path.exists(CHECKPOINT_PATH):
    print("Downloading MobileSAM checkpoint from Google Drive...")
    gdown.download(GDRIVE_URL, CHECKPOINT_PATH, quiet=False)

# ---------- Initialize MobileSAM ----------
model_type = "vit_t"  # tiny transformer
sam = sam_model_registry[model_type](checkpoint=CHECKPOINT_PATH)
sam.to("cpu")  # CPU ONLY
sam.eval()

mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,  # ↓ reduces memory further
    pred_iou_thresh=0.88,
    stability_score_thresh=0.95,
)

# ---------- Helper functions ----------
def decode_base64_image(img_b64: str):
    img_bytes = base64.b64decode(img_b64)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(img_pil)[:, :, ::-1]  # RGB → BGR

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
            hull = cv2.convexHull(cntr)
            cv2.drawContours(blackImg, [hull], -1, 255, 1)

    return blackImg

# ---------- Gradio function ----------
def segment_image_base64(img_b64: str):
    img = decode_base64_image(img_b64)
    masks = mask_generator.generate(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    bw_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for m in masks:
        bw_mask[m["segmentation"]] = 255

    processed_mask = apply_erosion_and_contours(bw_mask)
    mask_color = cv2.cvtColor(processed_mask, cv2.COLOR_GRAY2BGR)

    return encode_image_to_base64(mask_color)

# ---------- Gradio Interface ----------
demo = gr.Interface(
    fn=segment_image_base64,
    inputs=gr.Textbox(label="Base64 Image"),
    outputs=gr.Textbox(label="Mask Base64"),
)

# ---------- Launch ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        enable_queue=True  # optional, helps with multiple requests
    )
