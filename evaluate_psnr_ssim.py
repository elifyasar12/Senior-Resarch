from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from PIL import Image
import numpy as np
import os

# Set your paths
original_dir = "dataset/original/train/fracture"
compressed_dirs = {
    "JPEG 90": "dataset/q90/train/fracture",
    "JPEG 70": "dataset/q70/train/fracture",
    "JPEG 50": "dataset/q50/train/fracture"
}

def evaluate_compression(original_dir, compressed_dir):
    psnr_scores = []
    ssim_scores = []

    for filename in os.listdir(original_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue  # Skip non-image files

        orig_path = os.path.join(original_dir, filename)
        comp_path = os.path.join(compressed_dir, filename)
        if not os.path.exists(comp_path):
            continue

        # Load and resize images
        orig_img = Image.open(orig_path).convert("RGB").resize((224, 224))
        comp_img = Image.open(comp_path).convert("RGB").resize((224, 224))

        # Convert to arrays
        orig_arr = np.array(orig_img)
        comp_arr = np.array(comp_img)

        # Convert to grayscale for SSIM
        orig_gray = np.mean(orig_arr, axis=2)
        comp_gray = np.mean(comp_arr, axis=2)

        # Compute metrics
        psnr_scores.append(psnr(orig_arr, comp_arr))
        ssim_scores.append(ssim(orig_gray, comp_gray, data_range=255))

    return round(np.mean(psnr_scores), 2), round(np.mean(ssim_scores), 3)

# Run and print results
print("\n📊 PSNR & SSIM Evaluation:\n")
for label, path in compressed_dirs.items():
    psnr_val, ssim_val = evaluate_compression(original_dir, path)
    print(f"{label}: PSNR = {psnr_val} dB | SSIM = {ssim_val}")
