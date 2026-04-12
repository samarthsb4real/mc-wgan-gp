"""
Bug Condition Exploration Test for MC-WGAN-GP Conditioning Bypass

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3**

This test is EXPECTED TO FAIL on unfixed code.
Failure confirms the bug exists: generator ignores mask/class conditioning.

Three sub-tests:
  1. Cross-Class Similarity (SSIM) — images should differ across classes
  2. Mask Geometry Respect — defects should appear in masked regions
  3. Minority Class Distinctiveness — Fray vs Free should be classifiable

Expected counterexamples on unfixed code:
  - SSIM > 0.95 between different classes (images are identical)
  - Intensity ratio ~1.0 (no mask respect)
  - Classifier accuracy ~50% (classes indistinguishable)
"""

import os
import sys
import math
import glob
import random
import numpy as np
import cv2
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from skimage.metrics import structural_similarity as ssim_fn

# ---------------------------------------------------------------------------
# Ensure run.py classes are importable without executing main()
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

# We import only the classes/functions we need, not the full main() execution.
# run.py executes set_seed() and creates output dirs at module level — that's fine.
from run import (
    Config,
    MaskConditionedGenerator,
    MaskEncoder,
    _load_all_samples,
    get_stratified_split,
    set_seed,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHECKPOINT_PATH = "outputs/checkpoints/checkpoint_epoch_0100.pth"
SSIM_THRESHOLD  = 0.70   # Images for different classes should be < 0.70 similar
INTENSITY_RATIO_THRESHOLD = 1.5  # masked / unmasked pixel intensity ratio
CLASSIFIER_ACCURACY_THRESHOLD = 0.90  # 90% accuracy for minority class distinctiveness
FIXED_SEED = 42

set_seed(FIXED_SEED)

cfg = Config()
cfg.DEVICE = torch.device("cpu")  # Use CPU for reproducibility in tests


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_generator() -> MaskConditionedGenerator:
    """
    Load the generator from the latest checkpoint.
    Uses strict=False to handle architecture changes between old and new checkpoints.
    If the checkpoint architecture doesn't match (e.g., old pre-fix checkpoint),
    the generator will use freshly initialized weights for new layers.
    """
    gen = MaskConditionedGenerator(
        cfg.LATENT_DIM, cfg.MASK_LATENT_DIM,
        cfg.NUM_CLASSES, cfg.GEN_FEATURES,
        cfg.CHANNELS, cfg.IMAGE_SIZE,
    ).to(cfg.DEVICE)

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=cfg.DEVICE)
        missing, unexpected = gen.load_state_dict(ckpt["generator"], strict=False)
        if missing or unexpected:
            print(f"  NOTE: Checkpoint architecture mismatch (old pre-fix checkpoint).")
            print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
            print(f"  Using freshly initialized weights for new architecture layers.")
    else:
        print(f"  NOTE: Checkpoint not found at {CHECKPOINT_PATH}. Using fresh weights.")

    gen.eval()
    return gen


def load_real_masks_per_class() -> dict:
    """
    Load one real mask per class from the data directory.
    Returns dict: {label: mask_tensor (1, 1, 128, 128)}
    """
    all_samples = _load_all_samples(cfg.DATA_DIR, cfg.CLASS_NAMES)
    masks_by_class = {}

    for img_path, mask_path, label in all_samples:
        if label in masks_by_class:
            continue
        if mask_path and os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                               interpolation=cv2.INTER_NEAREST)
                mask_tensor = torch.from_numpy(
                    (m > 127).astype(np.float32)
                ).unsqueeze(0).unsqueeze(0)  # (1, 1, 128, 128)
                masks_by_class[label] = mask_tensor
        if len(masks_by_class) == cfg.NUM_CLASSES:
            break

    # Fallback: all-ones mask for any missing class
    for label in range(cfg.NUM_CLASSES):
        if label not in masks_by_class:
            masks_by_class[label] = torch.ones(1, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)

    return masks_by_class


def load_masks_for_class(label: int, n: int = 50) -> list:
    """Load up to n real masks for a given class label."""
    all_samples = _load_all_samples(cfg.DATA_DIR, cfg.CLASS_NAMES)
    class_samples = [(ip, mp, lbl) for ip, mp, lbl in all_samples if lbl == label]
    masks = []
    for img_path, mask_path, _ in class_samples[:n]:
        if mask_path and os.path.exists(mask_path):
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                               interpolation=cv2.INTER_NEAREST)
                t = torch.from_numpy((m > 127).astype(np.float32)).unsqueeze(0).unsqueeze(0)
                masks.append(t)
    # Pad with ones if not enough
    while len(masks) < n:
        masks.append(torch.ones(1, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
    return masks[:n]


def tensor_to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """Convert (1, 1, H, W) or (1, H, W) tensor in [-1,1] to uint8 HxW numpy."""
    arr = t.squeeze().detach().cpu().numpy()
    arr = (arr + 1.0) / 2.0  # [-1,1] → [0,1]
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def compute_ssim(img_a: torch.Tensor, img_b: torch.Tensor) -> float:
    """Compute SSIM between two (1,1,H,W) tensors in [-1,1]."""
    a = tensor_to_numpy_image(img_a)
    b = tensor_to_numpy_image(img_b)
    score, _ = ssim_fn(a, b, full=True, data_range=255)
    return float(score)


# ---------------------------------------------------------------------------
# Simple CNN classifier for minority class distinctiveness test
# ---------------------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Linear(32 * 4 * 4, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


def train_simple_classifier(images_a, images_b, epochs=10):
    """
    Train a simple CNN to distinguish class A (label=0) from class B (label=1).
    images_a, images_b: lists of (1,1,H,W) tensors in [-1,1]
    Returns accuracy on the training set (proxy for distinctiveness).
    """
    model = SimpleCNN(num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    imgs = torch.cat(images_a + images_b, dim=0)  # (N, 1, H, W)
    labels = torch.tensor(
        [0] * len(images_a) + [1] * len(images_b), dtype=torch.long
    )

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(imgs).argmax(dim=1)
    accuracy = (preds == labels).float().mean().item()
    return accuracy


# ===========================================================================
# TEST 1: Cross-Class Similarity (SSIM)
# **Validates: Requirements 1.1, 1.3, 2.1, 2.3**
# ===========================================================================

def test_cross_class_ssim_distinctiveness():
    """
    Property 1: Bug Condition — Cross-Class Similarity Test

    Generate images for all 6 classes using the SAME fixed noise vector
    but DIFFERENT real masks (one per class). Compute pairwise SSIM.

    EXPECTED ON UNFIXED CODE: SSIM > 0.95 for all pairs (images are identical).
    EXPECTED ON FIXED CODE:   SSIM < 0.70 for different classes.

    **Validates: Requirements 1.1, 1.3, 2.1, 2.3**
    """
    set_seed(FIXED_SEED)
    generator = load_generator()
    masks_by_class = load_real_masks_per_class()

    # Fixed noise vector — same for all classes
    fixed_noise = torch.randn(1, cfg.LATENT_DIM, generator=torch.Generator().manual_seed(FIXED_SEED))
    fixed_noise = fixed_noise.to(cfg.DEVICE)

    # Generate one image per class
    generated = {}
    with torch.no_grad():
        for label in range(cfg.NUM_CLASSES):
            mask = masks_by_class[label].to(cfg.DEVICE)
            label_tensor = torch.tensor([label], device=cfg.DEVICE)
            result = generator(fixed_noise, label_tensor, mask)
            # Fixed generator returns (image, pred_mask) tuple
            img = result[0] if isinstance(result, tuple) else result
            generated[label] = img.cpu()

    # Compute pairwise SSIM for all 15 class pairs
    failures = []
    ssim_values = {}
    class_names = cfg.DISPLAY_NAMES

    for i in range(cfg.NUM_CLASSES):
        for j in range(i + 1, cfg.NUM_CLASSES):
            s = compute_ssim(generated[i], generated[j])
            ssim_values[(i, j)] = s
            pair_name = f"{class_names[i]} vs {class_names[j]}"
            print(f"  SSIM({pair_name}): {s:.4f}")
            if s >= SSIM_THRESHOLD:
                failures.append(
                    f"COUNTEREXAMPLE: SSIM({pair_name}) = {s:.4f} >= {SSIM_THRESHOLD} "
                    f"(images are too similar — generator ignores class/mask conditioning)"
                )

    print(f"\nSSIM summary: min={min(ssim_values.values()):.4f}, "
          f"max={max(ssim_values.values()):.4f}, "
          f"mean={np.mean(list(ssim_values.values())):.4f}")

    if failures:
        failure_msg = "\n".join(failures)
        pytest.fail(
            f"BUG CONFIRMED: Generator produces visually identical outputs across classes.\n"
            f"{failure_msg}\n\n"
            f"This confirms Requirements 1.1 and 1.3: mask signal washout and "
            f"generator ignoring class conditioning."
        )


# ===========================================================================
# TEST 2: Mask Geometry Respect
# **Validates: Requirements 1.2, 1.3, 2.2, 2.3**
# ===========================================================================

def test_mask_geometry_respect():
    """
    Property 1: Bug Condition — Mask Geometry Respect Test

    For Blowhole class, generate images using same noise but different masks.
    Compute mask-weighted pixel intensity ratio (masked vs unmasked regions).

    EXPECTED ON UNFIXED CODE: Ratio ~1.0 (no spatial correlation with mask).
    EXPECTED ON FIXED CODE:   Ratio > 1.5 (defects appear in masked regions).

    **Validates: Requirements 1.2, 1.3, 2.2, 2.3**
    """
    set_seed(FIXED_SEED)
    generator = load_generator()

    # Use Blowhole (label=0) masks
    blowhole_masks = load_masks_for_class(label=0, n=5)

    fixed_noise = torch.randn(1, cfg.LATENT_DIM, generator=torch.Generator().manual_seed(FIXED_SEED))
    fixed_noise = fixed_noise.to(cfg.DEVICE)
    label_tensor = torch.tensor([0], device=cfg.DEVICE)

    failures = []
    ratios = []

    with torch.no_grad():
        for i, mask in enumerate(blowhole_masks):
            mask = mask.to(cfg.DEVICE)
            result = generator(fixed_noise, label_tensor, mask)
            # Fixed generator returns (image, pred_mask) tuple
            img = result[0] if isinstance(result, tuple) else result

            # Convert to [0, 1] range
            img_np = img.squeeze().cpu().numpy()
            img_np = (img_np + 1.0) / 2.0  # [-1,1] → [0,1]
            mask_np = mask.squeeze().cpu().numpy()  # binary {0, 1}

            masked_pixels   = img_np[mask_np > 0.5]
            unmasked_pixels = img_np[mask_np <= 0.5]

            if len(masked_pixels) == 0 or len(unmasked_pixels) == 0:
                # Degenerate mask — skip
                continue

            masked_mean   = float(np.mean(np.abs(masked_pixels)))
            unmasked_mean = float(np.mean(np.abs(unmasked_pixels)))

            if unmasked_mean < 1e-6:
                ratio = 1.0
            else:
                ratio = masked_mean / unmasked_mean

            ratios.append(ratio)
            print(f"  Mask {i}: masked_mean={masked_mean:.4f}, "
                  f"unmasked_mean={unmasked_mean:.4f}, ratio={ratio:.4f}")

            if ratio < INTENSITY_RATIO_THRESHOLD:
                failures.append(
                    f"COUNTEREXAMPLE: Mask {i} intensity ratio = {ratio:.4f} < "
                    f"{INTENSITY_RATIO_THRESHOLD} "
                    f"(defects do NOT appear preferentially in masked regions)"
                )

    if ratios:
        print(f"\nIntensity ratio summary: mean={np.mean(ratios):.4f}, "
              f"min={min(ratios):.4f}, max={max(ratios):.4f}")

    if failures:
        failure_msg = "\n".join(failures)
        pytest.fail(
            f"BUG CONFIRMED: Generator does not respect mask geometry.\n"
            f"{failure_msg}\n\n"
            f"This confirms Requirements 1.2 and 1.3: critic blindness to masks "
            f"and mask signal washout mean defects appear randomly, not in masked regions."
        )


# ===========================================================================
# TEST 3: Minority Class Distinctiveness
# **Validates: Requirements 1.4, 2.1**
# ===========================================================================

def test_minority_class_distinctiveness():
    """
    Property 1: Bug Condition — Minority Class Distinctiveness Test

    Generate 50 images each for Fray (label=3, minority: 27 samples) and
    Free (label=4, majority: 952 samples) using different noise vectors and
    real masks. Train a simple CNN classifier on these 100 synthetic images.

    EXPECTED ON UNFIXED CODE: Classifier accuracy ~50% (classes indistinguishable).
    EXPECTED ON FIXED CODE:   Classifier accuracy > 90%.

    **Validates: Requirements 1.4, 2.1**
    """
    set_seed(FIXED_SEED)
    generator = load_generator()

    N = 50  # images per class
    fray_label = 3   # Fray — minority class (27 samples)
    free_label = 4   # Free — majority class (952 samples)

    fray_masks = load_masks_for_class(fray_label, n=N)
    free_masks = load_masks_for_class(free_label, n=N)

    rng = torch.Generator()
    rng.manual_seed(FIXED_SEED)

    fray_images = []
    free_images = []

    with torch.no_grad():
        for i in range(N):
            noise = torch.randn(1, cfg.LATENT_DIM, generator=rng).to(cfg.DEVICE)

            # Fray
            mask_f = fray_masks[i].to(cfg.DEVICE)
            label_f = torch.tensor([fray_label], device=cfg.DEVICE)
            result_f = generator(noise, label_f, mask_f)
            img_f = (result_f[0] if isinstance(result_f, tuple) else result_f).cpu()
            fray_images.append(img_f)

            # Free (different noise)
            noise2 = torch.randn(1, cfg.LATENT_DIM, generator=rng).to(cfg.DEVICE)
            mask_r = free_masks[i].to(cfg.DEVICE)
            label_r = torch.tensor([free_label], device=cfg.DEVICE)
            result_r = generator(noise2, label_r, mask_r)
            img_r = (result_r[0] if isinstance(result_r, tuple) else result_r).cpu()
            free_images.append(img_r)

    accuracy = train_simple_classifier(fray_images, free_images, epochs=15)
    print(f"\nMinority class distinctiveness — CNN classifier accuracy: {accuracy:.4f}")

    if accuracy < CLASSIFIER_ACCURACY_THRESHOLD:
        pytest.fail(
            f"BUG CONFIRMED: Minority class Fray is indistinguishable from majority class Free.\n"
            f"COUNTEREXAMPLE: CNN classifier accuracy = {accuracy:.4f} < "
            f"{CLASSIFIER_ACCURACY_THRESHOLD}\n"
            f"(Expected ~50% on unfixed code — classes are visually identical)\n\n"
            f"This confirms Requirement 1.4: severe class imbalance provides insufficient "
            f"conditioning signal for minority classes. Generator defaults to a single mode."
        )


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BUG CONDITION EXPLORATION TEST — MC-WGAN-GP Conditioning Bypass")
    print("=" * 70)
    print("NOTE: These tests are EXPECTED TO FAIL on unfixed code.")
    print("      Failure confirms the bug exists.\n")
    pytest.main([__file__, "-v", "--tb=short", "-s"])
