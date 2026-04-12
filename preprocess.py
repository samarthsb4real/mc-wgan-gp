"""
preprocess.py — Offline preprocessing for the Magnetic Tile dataset.

What this does:
  1. Scans all (image, mask) pairs per class
  2. Pads each image/mask to square (preserving aspect ratio) then resizes to IMAGE_SIZE
  3. Binarizes masks (threshold=127) — handles JPEG compression noise in .png masks
  4. Saves preprocessed images as uint8 .png to data/processed/<CLASS>/imgs/
  5. Saves preprocessed masks as uint8 .png to data/processed/<CLASS>/masks/
  6. Computes and saves dataset statistics (mean, std, class counts) to data/processed/stats.json

Run once before training:
    python3 preprocess.py

After running, set USE_PREPROCESSED = True in run.py (or pass --preprocessed flag).
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR      = "data"
OUTPUT_DIR    = "data/processed"
IMAGE_SIZE    = 128
CLASS_NAMES   = ["MT_Blowhole", "MT_Break", "MT_Crack",
                 "MT_Fray", "MT_Free", "MT_Uneven"]
MASK_THRESHOLD = 127   # pixels above this → defect (1), below → background (0)


def pad_to_square(img, fill_value=0):
    """
    Pad image to square with fill_value, preserving aspect ratio.
    Shorter dimension gets equal padding on both sides.
    """
    h, w = img.shape[:2]
    if h == w:
        return img
    size = max(h, w)
    if len(img.shape) == 2:
        padded = np.full((size, size), fill_value, dtype=img.dtype)
    else:
        padded = np.full((size, size, img.shape[2]), fill_value, dtype=img.dtype)
    top  = (size - h) // 2
    left = (size - w) // 2
    padded[top:top+h, left:left+w] = img
    return padded


def preprocess_image(img_path, size=IMAGE_SIZE):
    """Load, pad to square, resize to size×size. Returns uint8 grayscale."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((size, size), dtype=np.uint8)
    img = pad_to_square(img, fill_value=int(img.mean()))  # pad with mean intensity
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    return img


def preprocess_mask(mask_path, size=IMAGE_SIZE):
    """
    Load mask, pad to square, resize, binarize.
    Returns uint8 with values 0 or 255.
    """
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros((size, size), dtype=np.uint8)
    mask = pad_to_square(mask, fill_value=0)  # pad masks with 0 (background)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    # Binarize — handles JPEG compression artifacts in .png masks
    mask = (mask > MASK_THRESHOLD).astype(np.uint8) * 255
    return mask


def main():
    print(f"Preprocessing Magnetic Tile dataset → {OUTPUT_DIR}/")
    print(f"Image size: {IMAGE_SIZE}×{IMAGE_SIZE} (aspect-ratio preserving pad + resize)\n")

    all_pixel_values = []
    class_counts = {}
    total = 0

    for label, cls in enumerate(CLASS_NAMES):
        src_dir  = Path(DATA_DIR) / cls / "Imgs"
        img_dir  = Path(OUTPUT_DIR) / cls / "imgs"
        mask_dir = Path(OUTPUT_DIR) / cls / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        jpg_files = sorted(src_dir.glob("*.jpg"))
        count = 0

        for jpg_path in jpg_files:
            png_path = jpg_path.with_suffix(".png")

            # Process image
            img = preprocess_image(jpg_path)
            cv2.imwrite(str(img_dir / jpg_path.name.replace(".jpg", ".png")), img)
            all_pixel_values.append(img.astype(np.float32).flatten() / 255.0)

            # Process mask
            if png_path.exists():
                mask = preprocess_mask(png_path)
            else:
                # No mask file — use all-zeros for Free class, all-ones otherwise
                if cls == "MT_Free":
                    mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
                else:
                    mask = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) * 255
            cv2.imwrite(str(mask_dir / jpg_path.name.replace(".jpg", ".png")), mask)
            count += 1

        class_counts[cls] = count
        total += count
        print(f"  {cls:15s}: {count:4d} images processed")

    # Compute dataset statistics
    all_pixels = np.concatenate(all_pixel_values)
    mean = float(all_pixels.mean())
    std  = float(all_pixels.std())

    # Compute class weights for classifier (inverse frequency)
    class_weights = {cls: total / (len(CLASS_NAMES) * cnt)
                     for cls, cnt in class_counts.items()}

    stats = {
        "image_size":    IMAGE_SIZE,
        "total_samples": total,
        "class_counts":  class_counts,
        "class_weights": class_weights,
        "pixel_mean":    mean,
        "pixel_std":     std,
        "normalize_mean": [mean],
        "normalize_std":  [std],
    }

    stats_path = Path(OUTPUT_DIR) / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset statistics:")
    print(f"  Total samples : {total}")
    print(f"  Pixel mean    : {mean:.4f}")
    print(f"  Pixel std     : {std:.4f}")
    print(f"\nClass weights (for weighted loss/sampler):")
    for cls, w in class_weights.items():
        print(f"  {cls:15s}: {w:.4f}  (n={class_counts[cls]})")
    print(f"\nStats saved → {stats_path}")
    print(f"Preprocessed data → {OUTPUT_DIR}/")
    print("\nDone. Set USE_PREPROCESSED=True in run.py before training.")


if __name__ == "__main__":
    main()
