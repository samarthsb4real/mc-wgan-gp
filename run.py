# =============================================================================
# MC-WGAN-GP: Mask-Conditioned WGAN-GP for Magnetic Tile Defect Synthesis
# Novelty: Segmentation mask used as a spatial structural prior for GAN conditioning
# =============================================================================
# Requirements:
#   pip install torch torchvision opencv-python matplotlib numpy
#   pip install scikit-learn tqdm tensorboard seaborn
# =============================================================================

import os
import random
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.models import resnet18, ResNet18_Weights, inception_v3
import scipy.linalg

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

class Config:
    # Paths
    DATA_DIR       = "data"
    PROCESSED_DIR  = "data/processed"   # preprocessed data (run preprocess.py first)
    OUTPUT_DIR     = "outputs"
    CHECKPOINT_DIR = "outputs/checkpoints"
    SAMPLE_DIR     = "outputs/samples"
    AUGMENTED_DIR  = "data/augmented"

    # Dataset
    IMAGE_SIZE   = 128
    NUM_CLASSES  = 6
    CLASS_NAMES  = ["MT_Blowhole", "MT_Break", "MT_Crack",
                    "MT_Fray", "MT_Free", "MT_Uneven"]
    DISPLAY_NAMES = ["Blowhole", "Break", "Crack", "Fray", "Free", "Uneven"]

    # Normalization stats from preprocess.py (actual dataset mean/std)
    # Used for classifier only — GAN uses standard [-1,1] normalization
    PIXEL_MEAN = 0.4312
    PIXEL_STD  = 0.1684

    # GAN normalization — standard [0.5]/[0.5] maps to [-1,1] matching Tanh output
    GAN_PIXEL_MEAN = 0.5
    GAN_PIXEL_STD  = 0.5

    # Class weights from preprocess.py (inverse frequency, for classifier loss)
    # Order: Blowhole, Break, Crack, Fray, Free, Uneven
    CLASS_WEIGHTS = [1.9478, 2.6353, 3.9298, 7.0000, 0.2353, 2.1748]

    # GAN Architecture
    LATENT_DIM      = 64    # Reduced: less freedom → less mode collapse on small data
    MASK_LATENT_DIM = 64    # Reduced to match LATENT_DIM
    GEN_FEATURES    = 16    # Reduced: start_ch = 16*2^4 = 256 → ~3M params (was 19M)
    CRITIC_FEATURES = 32    # Balanced with generator
    CHANNELS        = 1     # Grayscale

    # Training
    EPOCHS       = 100
    BATCH_SIZE   = 16
    LR_GEN       = 1e-4
    LR_CRITIC    = 2e-4    # Reduced TTUR ratio: 2× instead of 4× — critic was too dominant
    BETA1        = 0.0
    BETA2        = 0.9
    CRITIC_STEPS = 3        # Reduced from 5: fewer critic steps → generator gets more signal
    LAMBDA_GP    = 10
    GRAD_CLIP    = 1.0     # Generator gradient clipping
    SAVE_INTERVAL = 10

    # Augmentation
    SYNTH_PER_CLASS = 200

    # Classifier
    CLF_EPOCHS = 25
    CLF_LR     = 1e-4

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


cfg = Config()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


set_seed()

for d in [cfg.OUTPUT_DIR, cfg.CHECKPOINT_DIR, cfg.SAMPLE_DIR, cfg.AUGMENTED_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"Device        : {cfg.DEVICE}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU           : {torch.cuda.get_device_name(0)}")


# =============================================================================
# SECTION 2: DATASET — Paired (Image, Mask, Label)
# =============================================================================

def _load_all_samples(data_dir, class_names, processed_dir=None):
    """
    Scans dataset folders and returns a list of (img_path, mask_path, label).
    If processed_dir exists, loads from data/processed/<CLASS>/imgs/ and masks/.
    Otherwise falls back to raw data/CLASS/Imgs/*.jpg with paired .png masks.
    """
    # Use preprocessed data if available
    if processed_dir and Path(processed_dir).exists():
        all_samples = []
        for label, cls in enumerate(class_names):
            img_dir  = Path(processed_dir) / cls / "imgs"
            mask_dir = Path(processed_dir) / cls / "masks"
            if not img_dir.exists():
                print(f"WARNING: Preprocessed folder not found: {img_dir}")
                continue
            found = 0
            for img_path in sorted(img_dir.glob("*.png")):
                mask_path = mask_dir / img_path.name
                all_samples.append((
                    str(img_path),
                    str(mask_path) if mask_path.exists() else None,
                    label
                ))
                found += 1
            if found == 0:
                print(f"WARNING: No preprocessed images found in {img_dir}")
        return all_samples

    # Fallback: raw data
    all_samples = []
    for label, cls in enumerate(class_names):
        cls_dir = Path(data_dir) / cls / "Imgs"
        if not cls_dir.exists():
            cls_dir = Path(data_dir) / cls
        if not cls_dir.exists():
            print(f"WARNING: Folder not found: {cls_dir}")
            continue
        found = 0
        for img_path in sorted(cls_dir.glob("*.jpg")):
            mask_path = img_path.with_suffix(".png")
            all_samples.append((
                str(img_path),
                str(mask_path) if mask_path.exists() else None,
                label
            ))
            found += 1
        if found == 0:
            print(f"WARNING: No JPG images found in {cls_dir}")
    return all_samples


def get_stratified_split(all_samples, test_size=0.15, seed=42):
    """Stratified split — preserves class proportions even in minority classes."""
    labels = [s[2] for s in all_samples]
    idxs   = np.arange(len(all_samples))
    sss    = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(sss.split(idxs, labels))
    return train_idx.tolist(), val_idx.tolist()


class DefectDataset(Dataset):
    """
    Unified dataset for GAN training and classification.
    When loading preprocessed data (data/processed/), images are already
    128×128 grayscale PNGs and masks are already binary — no resize needed.

    Args:
        samples     : list of (img_path, mask_path_or_None, label)
        image_size  : spatial resize target (only applied to raw data)
        augment     : apply geometric augmentation
        return_mask : True → returns (img, mask, label); False → (img, label)
        extra_imgs  : list of (path, label) for synthetic images (no masks needed)
        pixel_mean  : dataset mean for normalization (default: from preprocess.py)
        pixel_std   : dataset std for normalization (default: from preprocess.py)
    """

    def __init__(self, samples, image_size=128, augment=False,
                 return_mask=True, extra_imgs=None,
                 pixel_mean=0.4312, pixel_std=0.1684):
        self.samples     = list(samples)
        self.image_size  = image_size
        self.augment     = augment
        self.return_mask = return_mask
        self.extra_imgs  = extra_imgs or []
        self.pixel_mean  = pixel_mean
        self.pixel_std   = pixel_std

        self.img_tf  = self._build_img_tf()
        self.mask_tf = self._build_mask_tf()

        total = len(self.samples) + len(self.extra_imgs)
        print(f"  Dataset: {len(self.samples)} real + {len(self.extra_imgs)} synthetic "
              f"= {total} total | augment={augment} | return_mask={return_mask}")

    def _build_img_tf(self):
        ops = [transforms.ToPILImage()]
        # Only resize if not already preprocessed to target size
        ops.append(transforms.Resize((self.image_size, self.image_size)))
        if self.augment:
            ops += [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
            ]
        ops += [
            transforms.Grayscale(1),
            transforms.ToTensor(),
            transforms.Normalize([self.pixel_mean], [self.pixel_std]),
        ]
        return transforms.Compose(ops)

    def _build_mask_tf(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])

    def get_all_labels(self):
        real_labels  = [s[2] for s in self.samples]
        extra_labels = [s[1] for s in self.extra_imgs]
        return real_labels + extra_labels

    def __len__(self):
        return len(self.samples) + len(self.extra_imgs)

    def __getitem__(self, idx):
        if idx < len(self.samples):
            img_path, mask_path, label = self.samples[idx]
            has_mask = True
        else:
            img_path, label = self.extra_imgs[idx - len(self.samples)]
            mask_path = None
            has_mask  = False

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        img_tensor = self.img_tf(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

        if not self.return_mask:
            return img_tensor, label

        # Load mask
        if has_mask and mask_path:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = None

        if mask is None:
            # No mask: use all-zeros (background) — correct for Free class
            mask_tensor = torch.zeros(1, self.image_size, self.image_size)
        else:
            mask_tensor = self.mask_tf(cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB))
            mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor, label


def print_class_dist(samples, display_names, header="Class Distribution"):
    counts = Counter([s[2] for s in samples])
    print(f"\n{header}:")
    for label, name in enumerate(display_names):
        print(f"   {name:12s}: {counts.get(label, 0)}")
    print()


def create_weighted_sampler(samples, cfg):
    """
    Creates a WeightedRandomSampler that oversamples minority classes to
    balance conditioning signal strength during GAN training.

    Inverse-frequency weights are computed per class, then multiplied by
    minority-class boost factors:
        Fray  (label=3) ×6
        Crack (label=2) ×3
        Break (label=1) ×2
    """
    MINORITY_MULTIPLIERS = {1: 2.0, 2: 3.0, 3: 6.0}  # Break, Crack, Fray

    counts = Counter([s[2] for s in samples])
    class_weights = {c: 1.0 / counts[c] for c in range(cfg.NUM_CLASSES)}

    # Apply minority class multipliers
    for cls, mult in MINORITY_MULTIPLIERS.items():
        if cls in class_weights:
            class_weights[cls] *= mult

    sample_weights = [class_weights[s[2]] for s in samples]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def get_dataloaders(all_samples, train_idx, val_idx, cfg):
    train_samples = [all_samples[i] for i in train_idx]
    val_samples   = [all_samples[i] for i in val_idx]

    print_class_dist(train_samples, cfg.DISPLAY_NAMES, "Train Split")
    print_class_dist(val_samples,   cfg.DISPLAY_NAMES, "Val Split")

    train_ds = DefectDataset(train_samples, cfg.IMAGE_SIZE, augment=True,  return_mask=True,
                             pixel_mean=cfg.GAN_PIXEL_MEAN, pixel_std=cfg.GAN_PIXEL_STD)
    val_ds   = DefectDataset(val_samples,   cfg.IMAGE_SIZE, augment=False, return_mask=True,
                             pixel_mean=cfg.GAN_PIXEL_MEAN, pixel_std=cfg.GAN_PIXEL_STD)

    weighted_sampler = create_weighted_sampler(train_samples, cfg)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              sampler=weighted_sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, train_samples, val_samples


# =============================================================================
# SECTION 3: MC-WGAN-GP ARCHITECTURE
# =============================================================================

class MaskEncoder(nn.Module):
    """
    Lightweight CNN that encodes a binary segmentation mask (1×H×W)
    into a fixed-size latent vector. Acts as the spatial structural prior
    injected into the generator.
    """
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16,  4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Linear(64 * 16, out_dim)

    def forward(self, mask):
        x = self.net(mask)
        return self.fc(x.view(x.size(0), -1))


class MaskConditionedGenerator(nn.Module):
    """
    MC-WGAN-GP Generator with multi-point mask injection.
    Conditioned on: noise vector + class label embedding + mask encoding.
    Mask encoding is injected at 5 points to prevent signal washout:
      Point 1: Input fusion (noise + label + mask concatenation)
      Point 2: Bottleneck (8×8 spatial, after FC projection)
      Point 3: Before decoder block at 16×16 input resolution
      Point 4: Before decoder block at 32×32 input resolution
      Point 5: Before decoder block at 64×64 input resolution

    Input  : noise (LATENT_DIM,) | labels (B,) | masks (B,1,H,W)
    Output : tuple of ((B, 1, IMAGE_SIZE, IMAGE_SIZE) in [-1, 1],
                        (B, 1, IMAGE_SIZE, IMAGE_SIZE) predicted mask in [0, 1])
    """
    # Number of decoder blocks that receive skip-connection mask injection
    # (points 3-5: blocks whose input is at 16×16, 32×32, 64×64)
    _SKIP_INJECT_BLOCKS = 3

    def __init__(self, latent_dim, mask_latent_dim, num_classes,
                 features, channels, image_size):
        super().__init__()
        self.init_size       = 8
        self.mask_latent_dim = mask_latent_dim
        self.num_ups         = int(math.log2(image_size)) - int(math.log2(self.init_size))
        self.start_ch        = features * (2 ** self.num_ups)

        self.label_emb    = nn.Embedding(num_classes, latent_dim)
        self.mask_encoder = MaskEncoder(out_dim=mask_latent_dim)

        # Point 1: input fusion (noise + label_enc + mask_enc)
        fused_dim = latent_dim + latent_dim + mask_latent_dim
        self.fc   = nn.Linear(fused_dim, self.start_ch * self.init_size * self.init_size)

        # Build decoder blocks individually so we can inject mask_enc between them.
        # Point 2: bottleneck injection adds mask_latent_dim channels before block 0.
        # Points 3-5: skip injections add mask_latent_dim channels before blocks 1, 2, 3.
        # Block 3 (→128×128) and the final conv receive no extra injection.
        self.decoder_blocks = nn.ModuleList()
        in_ch = self.start_ch + mask_latent_dim  # after bottleneck injection (point 2)
        for i in range(self.num_ups):
            out_ch = (self.start_ch >> (i + 1))  # start_ch / 2^(i+1)
            block = nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.decoder_blocks.append(block)
            # Blocks 0, 1, 2 produce 16×16, 32×32, 64×64 outputs.
            # The NEXT block (1, 2, 3) will receive skip injection, so add mask_latent_dim.
            if i < self._SKIP_INJECT_BLOCKS:
                in_ch = out_ch + mask_latent_dim  # skip injection for next block
            else:
                in_ch = out_ch

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_ch, channels, 3, 1, 1),
            nn.Tanh(),
        )

        # Segmentation head: takes raw bottleneck features (B, start_ch, 8, 8)
        self.seg_head = SegmentationHead(in_channels=self.start_ch)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _broadcast_mask(self, mask_enc, H, W):
        """Spatially broadcast mask_enc (B, C) → (B, C, H, W)."""
        return mask_enc.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)

    def forward(self, noise, labels, masks):
        label_enc = self.label_emb(labels)                           # (B, LATENT_DIM)
        mask_enc  = self.mask_encoder(masks)                         # (B, MASK_LATENT_DIM)

        # Point 1: input fusion (existing behaviour preserved)
        fused = torch.cat([noise, label_enc, mask_enc], dim=1)       # (B, fused_dim)
        x = self.fc(fused).view(-1, self.start_ch, self.init_size, self.init_size)
        # x: (B, start_ch, 8, 8)

        # Pass raw bottleneck to segmentation head before any injection
        pred_mask = self.seg_head(x)                                  # (B, 1, 128, 128)

        # Point 2: bottleneck injection at 8×8
        x = torch.cat([x, self._broadcast_mask(mask_enc, x.size(2), x.size(3))], dim=1)

        # Decoder blocks with skip injections at points 3-5
        for i, block in enumerate(self.decoder_blocks):
            x = block(x)
            # After blocks 0, 1, 2 (outputs at 16×16, 32×32, 64×64) inject mask_enc
            if i < self._SKIP_INJECT_BLOCKS:
                x = torch.cat([x, self._broadcast_mask(mask_enc, x.size(2), x.size(3))], dim=1)

        generated = self.final_conv(x)                               # (B, 1, 128, 128)
        return generated, pred_mask


class SegmentationHead(nn.Module):
    """
    Auxiliary segmentation head for mask reconstruction.
    Takes bottleneck features (B, in_channels, 8, 8) and upsamples to
    predict a binary mask (B, 1, 128, 128) with values in [0, 1].

    Architecture: 4 ConvTranspose2d layers (8→16→32→64→128 spatial),
    each followed by BatchNorm2d + LeakyReLU(0.2), then a final
    Conv2d(features, 1, 3, 1, 1) + Sigmoid.
    """
    def __init__(self, in_channels):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(4):
            out_ch = ch // 2
            layers += [
                nn.ConvTranspose2d(ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = out_ch
        layers += [nn.Conv2d(ch, 1, 3, 1, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class MaskConditionedCritic(nn.Module):
    """
    MC-WGAN-GP Critic.
    Receives image + binary mask as a 2-channel input.
    Label conditioning is handled by the generator only; the critic sees
    masks directly to penalise mask-inconsistent generations.
    Uses InstanceNorm (required for WGAN-GP — no BatchNorm in critic).

    Input  : img (B,1,H,W) | labels (B,) [unused internally] | masks (B,1,H,W)
    Output : Wasserstein score (B, 1) — no sigmoid
    """
    def __init__(self, num_classes, features, image_size):
        super().__init__()
        self.image_size = image_size

        num_downs = int(math.log2(image_size)) - 2   # 128→5 downs→4×4 spatial
        layers, in_ch, out_ch = [], 2, features      # 2ch: img + mask
        for i in range(num_downs):
            layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False))
            if i > 0:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch  = out_ch
            out_ch = min(out_ch * 2, features * 16)

        self.net         = nn.Sequential(*layers)
        self.final_ch    = in_ch
        self.fc          = nn.Linear(self.final_ch * 4 * 4, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img, labels, masks):
        x = torch.cat([img, masks], dim=1)   # (B, 2, H, W)
        x = self.net(x).view(x.size(0), -1)
        return self.fc(x)


# =============================================================================
# SECTION 4: GRADIENT PENALTY
# =============================================================================

def compute_gradient_penalty(critic, real, fake, labels, masks, device):
    B     = real.size(0)
    alpha = torch.rand(B, 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score  = critic(interp, labels, masks)
    grads  = torch.autograd.grad(
        outputs=score, inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True, only_inputs=True,
    )[0]
    grads = grads.view(B, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


# =============================================================================
# SECTION 5: TRAINING LOOP
# =============================================================================

def _sample_fixed_masks(train_samples, cfg):
    """
    For each class, grab one real mask to use in fixed visualisation grid.
    Repeats 4× so grid shows 4 rows × NUM_CLASSES cols.
    """
    per_class = {}
    for _, mask_path, label in train_samples:
        if label in per_class:
            continue
        if mask_path:
            m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                               interpolation=cv2.INTER_NEAREST)
                per_class[label] = torch.from_numpy(
                    (m > 127).astype(np.float32)
                ).unsqueeze(0)
        if label not in per_class:
            per_class[label] = torch.ones(1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        if len(per_class) == cfg.NUM_CLASSES:
            break

    tiles = []
    for _ in range(4):
        for lbl in range(cfg.NUM_CLASSES):
            tiles.append(per_class.get(lbl, torch.ones(1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)))
    return torch.stack(tiles).to(cfg.DEVICE)


def train_mc_wgan_gp(all_samples, train_idx, val_idx, cfg):
    print("\n" + "=" * 60)
    print("  MC-WGAN-GP Training  (Mask-Conditioned)")
    print("=" * 60)

    train_loader, val_loader, train_samples, val_samples = get_dataloaders(
        all_samples, train_idx, val_idx, cfg
    )

    generator = MaskConditionedGenerator(
        cfg.LATENT_DIM, cfg.MASK_LATENT_DIM,
        cfg.NUM_CLASSES, cfg.GEN_FEATURES,
        cfg.CHANNELS, cfg.IMAGE_SIZE,
    ).to(cfg.DEVICE)

    critic = MaskConditionedCritic(
        cfg.NUM_CLASSES, cfg.CRITIC_FEATURES, cfg.IMAGE_SIZE,
    ).to(cfg.DEVICE)

    print(f"\nGenerator params : {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Critic params    : {sum(p.numel() for p in critic.parameters()):,}")

    opt_gen    = optim.Adam(generator.parameters(), lr=cfg.LR_GEN,
                            betas=(cfg.BETA1, cfg.BETA2))
    opt_critic = optim.Adam(critic.parameters(),    lr=cfg.LR_CRITIC,
                            betas=(cfg.BETA1, cfg.BETA2))

    writer = SummaryWriter(log_dir="outputs/tensorboard")

    fixed_noise  = torch.randn(cfg.NUM_CLASSES * 4, cfg.LATENT_DIM, device=cfg.DEVICE)
    fixed_labels = torch.tensor(list(range(cfg.NUM_CLASSES)) * 4, device=cfg.DEVICE)
    fixed_masks  = _sample_fixed_masks(train_samples, cfg)

    lambda_mask = 2.0   # Balanced: mask accountability without drowning adversarial signal

    history = {"critic_loss": [], "gen_loss": [], "w_distance": []}
    global_step = 0

    # Track best checkpoint by highest W distance (higher W = stronger generator)
    best_w_dist = float('-inf')

    for epoch in range(1, cfg.EPOCHS + 1):
        generator.train()
        critic.train()
        ep_c, ep_g, ep_w = [], [], []

        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{cfg.EPOCHS}]", leave=False)

        for real_imgs, masks, labels in loop:
            real_imgs = real_imgs.to(cfg.DEVICE)
            masks     = masks.to(cfg.DEVICE)
            labels    = labels.to(cfg.DEVICE)
            B         = real_imgs.size(0)

            # ── Critic ────────────────────────────────────────────
            for _ in range(cfg.CRITIC_STEPS):
                noise               = torch.randn(B, cfg.LATENT_DIM, device=cfg.DEVICE)
                fake_imgs, _        = generator(noise, labels, masks)
                fake_imgs           = fake_imgs.detach()

                real_score = critic(real_imgs, labels, masks)
                fake_score = critic(fake_imgs, labels, masks)
                gp         = compute_gradient_penalty(
                    critic, real_imgs, fake_imgs, labels, masks, cfg.DEVICE
                )
                c_loss = fake_score.mean() - real_score.mean() + cfg.LAMBDA_GP * gp

                opt_critic.zero_grad()
                c_loss.backward()
                opt_critic.step()

            ep_c.append(c_loss.item())

            # ── Generator ─────────────────────────────────────────
            noise                  = torch.randn(B, cfg.LATENT_DIM, device=cfg.DEVICE)
            fake_imgs, pred_masks  = generator(noise, labels, masks)
            fake_score             = critic(fake_imgs, labels, masks)
            bce_loss               = nn.BCELoss()(pred_masks, masks)
            g_loss                 = -fake_score.mean() + lambda_mask * bce_loss

            opt_gen.zero_grad()
            g_loss.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), cfg.GRAD_CLIP)
            opt_gen.step()

            ep_g.append(g_loss.item())
            w = real_score.mean().item() - fake_score.mean().item()
            ep_w.append(w)

            writer.add_scalar("Loss/Critic",           c_loss.item(), global_step)
            writer.add_scalar("Loss/Generator",        g_loss.item(), global_step)
            writer.add_scalar("Loss/MaskReconstruction", bce_loss.item(), global_step)
            writer.add_scalar("W_Distance",            w,             global_step)
            global_step += 1

            loop.set_postfix(C=f"{c_loss.item():.3f}",
                             G=f"{g_loss.item():.3f}",
                             W=f"{w:.3f}")

        avg_c, avg_g, avg_w = np.mean(ep_c), np.mean(ep_g), np.mean(ep_w)
        history["critic_loss"].append(avg_c)
        history["gen_loss"].append(avg_g)
        history["w_distance"].append(avg_w)

        print(f"Epoch {epoch:4d}/{cfg.EPOCHS} | "
              f"C: {avg_c:8.4f} | G: {avg_g:8.4f} | W: {avg_w:8.4f}")

        if epoch % cfg.SAVE_INTERVAL == 0 or epoch == 1:
            save_gan_samples(generator, fixed_noise, fixed_labels, fixed_masks,
                             epoch, cfg)
            save_checkpoint(generator, critic, opt_gen, opt_critic, epoch, cfg)

        # Save best_checkpoint.pth at the epoch with highest W distance.
        # Higher W distance = generator is stronger relative to critic.
        # W distance peaks around the middle of training before the critic
        # catches up — that's when generated images are highest quality.
        if avg_w > best_w_dist:
            best_w_dist = avg_w
            torch.save({
                "epoch":            epoch,
                "generator":        generator.state_dict(),
                "critic":           critic.state_dict(),
                "opt_gen":          opt_gen.state_dict(),
                "opt_critic":       opt_critic.state_dict(),
                "segmentation_head": generator.seg_head.state_dict(),
            }, f"{cfg.CHECKPOINT_DIR}/best_checkpoint.pth")
            print(f"   ✓ Best checkpoint saved at epoch {epoch} (W={best_w_dist:.4f})")

    writer.close()
    print("\nGAN Training Complete.")
    plot_training_history(history, cfg)
    return generator, critic, train_samples, val_samples


# =============================================================================
# SECTION 6: UTILITIES — SAVING & VISUALIZATION
# =============================================================================

def save_gan_samples(generator, noise, labels, masks, epoch, cfg):
    generator.eval()
    with torch.no_grad():
        fake, pred_masks = generator(noise, labels, masks)
    generator.train()

    img_grid  = vutils.make_grid(fake,       nrow=cfg.NUM_CLASSES, normalize=True, padding=2)
    mask_grid = vutils.make_grid(pred_masks, nrow=cfg.NUM_CLASSES, normalize=True, padding=2)

    img_np  = img_grid.permute(1, 2, 0).cpu().numpy()
    mask_np = mask_grid.permute(1, 2, 0).cpu().numpy()

    fig = plt.figure(figsize=(14, 14), constrained_layout=True)
    gs  = fig.add_gridspec(3, 1, height_ratios=[1, 9, 9])
    lax  = fig.add_subplot(gs[0])
    iax  = fig.add_subplot(gs[1])
    max_ = fig.add_subplot(gs[2])

    iax.imshow(img_np, cmap="gray")
    iax.axis("off")
    iax.set_title("Generated Images", fontsize=13, fontweight="bold")

    max_.imshow(mask_np, cmap="gray")
    max_.axis("off")
    max_.set_title("Predicted Masks", fontsize=13, fontweight="bold")

    lax.set_xlim(0, cfg.NUM_CLASSES)
    lax.set_ylim(0, 1)
    lax.axis("off")
    for i, name in enumerate(cfg.DISPLAY_NAMES):
        lax.text(i + 0.5, 0.5, name, ha="center", va="center",
                 fontsize=10, fontweight="bold")

    fig.suptitle(f"MC-WGAN-GP Generated Samples — Epoch {epoch}",
                 fontsize=14, fontweight="bold")

    path = f"{cfg.SAMPLE_DIR}/epoch_{epoch:04d}.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"   Samples → {path}")


def save_checkpoint(gen, critic, opt_gen, opt_critic, epoch, cfg):
    torch.save({
        "epoch":            epoch,
        "generator":        gen.state_dict(),
        "critic":           critic.state_dict(),
        "opt_gen":          opt_gen.state_dict(),
        "opt_critic":       opt_critic.state_dict(),
        "segmentation_head": gen.seg_head.state_dict(),
    }, f"{cfg.CHECKPOINT_DIR}/checkpoint_epoch_{epoch:04d}.pth")


def load_checkpoint(gen, critic, opt_gen, opt_critic, path, cfg):
    ckpt = torch.load(path, map_location=cfg.DEVICE, weights_only=False)
    gen.load_state_dict(ckpt["generator"])
    critic.load_state_dict(ckpt["critic"])
    opt_gen.load_state_dict(ckpt["opt_gen"])
    opt_critic.load_state_dict(ckpt["opt_critic"])
    if "segmentation_head" in ckpt:
        gen.seg_head.load_state_dict(ckpt["segmentation_head"])
    else:
        print("WARNING: segmentation_head not found in checkpoint, skipping")
    print(f"Checkpoint loaded from epoch {ckpt['epoch']}")
    return ckpt["epoch"]


def plot_training_history(history, cfg):
    keys    = ["critic_loss", "gen_loss", "w_distance"]
    colors  = ["crimson", "steelblue", "darkorange"]
    titles  = ["Critic Loss", "Generator Loss", "Wasserstein Distance"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, key, color, title in zip(axes, keys, colors, titles):
        ax.plot(history[key], color=color, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.OUTPUT_DIR}/training_curves.png", dpi=120)
    plt.close()
    print(f"Training curves → {cfg.OUTPUT_DIR}/training_curves.png")


# =============================================================================
# SECTION 7: SYNTHETIC IMAGE GENERATION
# =============================================================================

def generate_synthetic_images(generator, train_samples, cfg, n_per_class=None,
                              target_classes=None):
    """
    Generates n_per_class synthetic images for target_classes only, conditioned
    on randomly sampled real segmentation masks from each class.

    Args:
        generator     : trained MaskConditionedGenerator
        train_samples : list of (img_path, mask_path_or_None, label)
        cfg           : Config object
        n_per_class   : number of images to generate per class (default: cfg.SYNTH_PER_CLASS)
        target_classes: list of class indices to generate for
                        (default: [1, 2, 3] — Break, Crack, Fray minority classes only)

    Output: data/augmented/<CLASS_NAME>/synth_XXXXX.png
    Returns: dict mapping class index to list of generated image paths
    """
    if n_per_class is None:
        n_per_class = cfg.SYNTH_PER_CLASS

    if target_classes is None:
        target_classes = [1, 2, 3]  # Break, Crack, Fray — minority classes only

    target_names = [cfg.DISPLAY_NAMES[c] for c in target_classes]
    print(f"Generating synthetic images for minority classes only: {', '.join(target_names)}")

    best_ckpt = os.path.join(cfg.CHECKPOINT_DIR, "best_checkpoint.pth")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=cfg.DEVICE, weights_only=False)
        generator.load_state_dict(ckpt["generator"])
        print(f"Loaded best checkpoint from epoch {ckpt['epoch']}")

    generator.eval()
    print(f"  Generating {n_per_class} images per class for classes: {target_classes}")

    # Group real masks by class
    masks_by_class = {i: [] for i in range(cfg.NUM_CLASSES)}
    for _, mask_path, label in train_samples:
        if mask_path:
            masks_by_class[label].append(mask_path)

    synth_paths_by_class = {}

    for cls_idx in target_classes:
        class_name = cfg.CLASS_NAMES[cls_idx]
        class_dir  = Path(cfg.AUGMENTED_DIR) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        available = masks_by_class[cls_idx]
        generated = 0
        saved_paths = []

        with torch.no_grad():
            while generated < n_per_class:
                batch  = min(32, n_per_class - generated)
                noise  = torch.randn(batch, cfg.LATENT_DIM, device=cfg.DEVICE)
                labels = torch.full((batch,), cls_idx, dtype=torch.long,
                                    device=cfg.DEVICE)

                # Sample random real masks for this class
                mask_list = []
                for _ in range(batch):
                    mp = random.choice(available) if available else None
                    if mp:
                        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                        m = cv2.resize(m, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                                       interpolation=cv2.INTER_NEAREST)
                        m = torch.from_numpy(
                            (m > 127).astype(np.float32)
                        ).unsqueeze(0)
                    else:
                        m = torch.ones(1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
                    mask_list.append(m)

                mask_tensor = torch.stack(mask_list).to(cfg.DEVICE)
                imgs, _ = generator(noise, labels, mask_tensor)
                imgs = ((imgs + 1) / 2 * 255).clamp(0, 255)
                imgs = imgs.squeeze(1).cpu().numpy().astype(np.uint8)

                for i, img in enumerate(imgs):
                    out_path = class_dir / f"synth_{generated + i:05d}.png"
                    cv2.imwrite(str(out_path), img)
                    saved_paths.append(out_path)

                generated += batch

        synth_paths_by_class[cls_idx] = saved_paths
        disp = cfg.DISPLAY_NAMES[cls_idx]
        print(f"   {disp:12s}: {n_per_class} images → {class_dir}")

    print(f"\nMinority class synthetic images saved to: {cfg.AUGMENTED_DIR}/")
    return synth_paths_by_class


def compute_per_class_fid(real_loader, synth_paths_by_class, cfg):
    """
    Compute per-class FID between real and synthetic image distributions.
    Uses InceptionV3 pretrained on ImageNet, adapted for 1-channel grayscale input.

    Args:
        real_loader          : train DataLoader (yields (img, mask, label) or (img, label))
        synth_paths_by_class : dict mapping class index → list of synthetic image paths
        cfg                  : Config object

    Returns:
        filtered dict of synth_paths_by_class containing only classes where FID < threshold
    """
    fid_threshold = 300.0

    # With small minority classes (27-98 samples), 2048-dim InceptionV3 features
    # produce degenerate covariance matrices. Use PCA to reduce feature dimension
    # before FID computation for numerical stability.

    # Load InceptionV3 and adapt for 1-channel grayscale input
    inception = inception_v3(weights="IMAGENET1K_V1", transform_input=False)
    # Average RGB weights in first conv layer to handle 1-channel input
    orig_weight = inception.Conv2d_1a_3x3.conv.weight.data.mean(dim=1, keepdim=True)
    inception.Conv2d_1a_3x3.conv = nn.Conv2d(
        1, inception.Conv2d_1a_3x3.conv.out_channels,
        kernel_size=inception.Conv2d_1a_3x3.conv.kernel_size,
        stride=inception.Conv2d_1a_3x3.conv.stride,
        padding=inception.Conv2d_1a_3x3.conv.padding,
        bias=False,
    )
    inception.Conv2d_1a_3x3.conv.weight.data = orig_weight
    inception = inception.eval().to(cfg.DEVICE)

    # Hook to extract pool3 features (2048-dim)
    pool3_features = {}

    def _hook(module, input, output):
        pool3_features["feat"] = output.squeeze(-1).squeeze(-1)

    inception.avgpool.register_forward_hook(_hook)

    resize_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    def _extract_features_from_loader(loader, class_label):
        """Extract InceptionV3 features for images of a specific class."""
        feats = []
        with torch.no_grad():
            for batch in loader:
                # Support both (img, mask, label) and (img, label) formats
                if len(batch) == 3:
                    imgs, _, labels = batch
                else:
                    imgs, labels = batch
                mask = labels == class_label
                if not mask.any():
                    continue
                imgs = imgs[mask]
                # Resize to 299×299 for InceptionV3
                resized = []
                for img in imgs:
                    # img is (1, H, W) normalized to [-1, 1]; convert to uint8 for ToPILImage
                    img_np = ((img.squeeze(0).cpu().numpy() + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
                    resized.append(resize_tf(img_np))
                resized = torch.stack(resized).to(cfg.DEVICE)
                inception(resized)
                feats.append(pool3_features["feat"].cpu().numpy())
        if not feats:
            return None
        return np.concatenate(feats, axis=0)

    def _extract_features_from_paths(paths):
        """Extract InceptionV3 features for synthetic images from file paths."""
        feats = []
        batch_size = 32
        for i in range(0, len(paths), batch_size):
            batch_paths = paths[i:i + batch_size]
            imgs = []
            for p in batch_paths:
                img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if img is None:
                    img = np.zeros((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), dtype=np.uint8)
                imgs.append(resize_tf(img))
            imgs = torch.stack(imgs).to(cfg.DEVICE)
            with torch.no_grad():
                inception(imgs)
            feats.append(pool3_features["feat"].cpu().numpy())
        if not feats:
            return None
        return np.concatenate(feats, axis=0)

    def _compute_fid(feat_real, feat_synth):
        """Compute FID between two feature sets using mean and covariance.
        Uses PCA to reduce dimensionality when sample count is small,
        preventing degenerate covariance matrices.
        """
        n_real, n_synth = len(feat_real), len(feat_synth)
        n_min = min(n_real, n_synth)

        # Reduce to safe dimensionality: at most n_samples - 1 dims
        # This prevents singular covariance matrices with small datasets
        max_dims = min(n_min - 1, 64)  # cap at 64 dims for stability
        if feat_real.shape[1] > max_dims and max_dims >= 2:
            pca = PCA(n_components=max_dims, random_state=42)
            combined = np.concatenate([feat_real, feat_synth], axis=0)
            pca.fit(combined)
            feat_real  = pca.transform(feat_real)
            feat_synth = pca.transform(feat_synth)

        mu1, sigma1 = feat_real.mean(axis=0), np.cov(feat_real, rowvar=False)
        mu2, sigma2 = feat_synth.mean(axis=0), np.cov(feat_synth, rowvar=False)

        # Ensure 2D covariance matrices (edge case: single feature)
        if sigma1.ndim == 0:
            sigma1 = np.array([[sigma1]])
        if sigma2.ndim == 0:
            sigma2 = np.array([[sigma2]])

        diff = mu1 - mu2
        # Add small epsilon to diagonal for numerical stability
        eps = 1e-6
        sigma1 += np.eye(sigma1.shape[0]) * eps
        sigma2 += np.eye(sigma2.shape[0]) * eps
        # Matrix square root via scipy
        covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
        if np.iscomplexobj(covmean):
            if np.abs(covmean.imag).max() > 1e-2:
                return float('inf')
            covmean = covmean.real
        fid = float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
        return max(fid, 0.0)

    print("\nComputing per-class FID for quality gating...")
    filtered = {}

    for cls_idx, paths in synth_paths_by_class.items():
        class_name = cfg.CLASS_NAMES[cls_idx]

        # Extract real features for this class
        feat_real = _extract_features_from_loader(real_loader, cls_idx)
        if feat_real is None or len(feat_real) < 2:
            print(f"Class {class_name}: too few real images — SKIP")
            continue

        # Extract synthetic features
        feat_synth = _extract_features_from_paths(paths)
        if feat_synth is None or len(feat_synth) < 2:
            print(f"Class {class_name}: too few synthetic images — SKIP")
            continue

        fid = _compute_fid(feat_real, feat_synth)
        passed = fid < fid_threshold
        status = "PASS" if passed else "REJECT"
        print(f"Class {class_name}: FID={fid:.2f} {status}")

        if passed:
            filtered[cls_idx] = paths

    print(f"\nFID gating: {len(filtered)}/{len(synth_paths_by_class)} classes passed "
          f"(threshold={fid_threshold})")
    return filtered


# =============================================================================
# SECTION 8: CLASSIFIER — Fine-tuned ResNet-18
# =============================================================================

class ResNet18Classifier(nn.Module):
    """
    Pre-trained ResNet-18 adapted for:
    - 1-channel grayscale input (first conv layer weights averaged)
    - Custom number of output classes (FC replaced)
    """
    def __init__(self, num_classes):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Adapt first conv: 3-ch → 1-ch
        w = base.conv1.weight.data.mean(dim=1, keepdim=True)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        base.conv1.weight.data = w
        # Replace classifier head
        base.fc = nn.Linear(512, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)


def _make_clf_loaders(train_samples, val_samples, synth_paths_by_class, cfg):
    """
    Build (img, label) loaders for classifier.
    synth_paths_by_class: dict {label: [list of synth png paths]} or None for baseline.
    """
    extra = []
    if synth_paths_by_class:
        for label, paths in synth_paths_by_class.items():
            for p in paths:
                extra.append((str(p), label))

    train_ds = DefectDataset(train_samples, cfg.IMAGE_SIZE,
                             augment=True, return_mask=False,
                             extra_imgs=extra,
                             pixel_mean=cfg.PIXEL_MEAN, pixel_std=cfg.PIXEL_STD)
    val_ds   = DefectDataset(val_samples,   cfg.IMAGE_SIZE,
                             augment=False, return_mask=False,
                             pixel_mean=cfg.PIXEL_MEAN, pixel_std=cfg.PIXEL_STD)

    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader


def _collect_synth_paths(cfg):
    """Collect all generated synthetic image paths grouped by class label."""
    out = {}
    for label, class_name in enumerate(cfg.CLASS_NAMES):
        class_dir = Path(cfg.AUGMENTED_DIR) / class_name
        if class_dir.exists():
            paths = list(class_dir.glob("*.png"))
            if paths:
                out[label] = paths
    return out


def train_classifier(train_loader, val_loader, cfg, tag="Baseline"):
    print(f"\n{'=' * 60}")
    print(f"  Training Classifier — {tag}")
    print(f"{'=' * 60}")

    model     = ResNet18Classifier(cfg.NUM_CLASSES).to(cfg.DEVICE)
    # Class-weighted loss to handle imbalance (Free=952 vs Fray=27)
    weights   = torch.tensor(cfg.CLASS_WEIGHTS, dtype=torch.float32).to(cfg.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=cfg.CLF_LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.CLF_EPOCHS)
    best_acc  = 0.0

    for epoch in range(1, cfg.CLF_EPOCHS + 1):
        model.train()
        correct, total = 0, 0

        for imgs, labels in tqdm(train_loader,
                                 desc=f"  Epoch {epoch}/{cfg.CLF_EPOCHS}",
                                 leave=False):
            imgs, labels = imgs.to(cfg.DEVICE), labels.to(cfg.DEVICE)
            out  = model(imgs)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == labels).sum().item()
            total   += imgs.size(0)

        scheduler.step()
        train_acc        = correct / total * 100
        val_acc, report  = evaluate_classifier(model, val_loader, cfg)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       f"{cfg.OUTPUT_DIR}/best_clf_{tag.lower().replace(' ','_')}.pth")

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train: {train_acc:.2f}% | "
                  f"Val: {val_acc:.2f}% | Best: {best_acc:.2f}%")

    print(f"\n  Best Val Accuracy ({tag}): {best_acc:.2f}%")
    print(f"\n  Classification Report:\n{report}")
    return model, best_acc


def evaluate_classifier(model, loader, cfg):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(cfg.DEVICE)).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    acc    = np.mean(np.array(all_preds) == np.array(all_labels)) * 100
    report = classification_report(all_labels, all_preds,
                                   target_names=cfg.DISPLAY_NAMES,
                                   zero_division=0)
    return acc, report


def plot_confusion_matrix(model, loader, cfg, tag=""):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            preds = model(imgs.to(cfg.DEVICE)).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=cfg.DISPLAY_NAMES,
                yticklabels=cfg.DISPLAY_NAMES, ax=ax)
    ax.set_title(f"Confusion Matrix — {tag}", fontsize=13, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    save_path = f"{cfg.OUTPUT_DIR}/confusion_matrix_{tag.replace(' ', '_')}.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Confusion matrix → {save_path}")


# =============================================================================
# SECTION 9: RESULTS COMPARISON
# =============================================================================

def compare_results(baseline_acc, augmented_acc, cfg):
    labels = ["Baseline\n(Real Data Only)", "MC-WGAN-GP\nAugmented"]
    accs   = [baseline_acc, augmented_acc]
    colors = ["#e74c3c", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, accs, color=colors, width=0.4,
                  edgecolor="black", linewidth=0.8)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{acc:.2f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold")
    delta = augmented_acc - baseline_acc
    ax.set_title(f"Baseline vs MC-WGAN-GP Augmented\n(Δ = {delta:+.2f}%)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{cfg.OUTPUT_DIR}/results_comparison.png", dpi=120)
    plt.close()
    print(f"Results comparison → {cfg.OUTPUT_DIR}/results_comparison.png")


# =============================================================================
# SECTION 10: MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("  MC-WGAN-GP: Mask-Conditioned Defect Synthesis")
    print("  Magnetic Tile Surface Defect Dataset")
    print("=" * 60)

    # ── Data split (computed once — shared by GAN and classifier) ───
    print("\nScanning dataset...")
    all_samples = _load_all_samples(cfg.DATA_DIR, cfg.CLASS_NAMES, cfg.PROCESSED_DIR)
    print(f"Total real image pairs found: {len(all_samples)}")
    print_class_dist(all_samples, cfg.DISPLAY_NAMES, "Full Dataset")

    train_idx, val_idx = get_stratified_split(all_samples)
    train_samples = [all_samples[i] for i in train_idx]
    val_samples   = [all_samples[i] for i in val_idx]

    # ── Phase 1: Train MC-WGAN-GP ────────────────────────────────────
    generator, critic, train_samples, val_samples = train_mc_wgan_gp(
        all_samples, train_idx, val_idx, cfg
    )

    # ── Phase 2: Generate Synthetic Images ───────────────────────────
    synth_paths = generate_synthetic_images(generator, train_samples, cfg)

    # ── Phase 2b: FID Quality Gating ─────────────────────────────────
    # Build a real-image loader for FID computation (train split, with masks)
    fid_real_loader, _, _, _ = get_dataloaders(all_samples, train_idx, val_idx, cfg)
    synth_paths = compute_per_class_fid(fid_real_loader, synth_paths, cfg)

    # ── Phase 3: Baseline Classifier (real data only) ─────────────────
    print("\nPhase 3: Baseline Classifier (real data only)")
    base_train_loader, base_val_loader = _make_clf_loaders(
        train_samples, val_samples, None, cfg
    )
    baseline_model, baseline_acc = train_classifier(
        base_train_loader, base_val_loader, cfg, tag="Baseline"
    )
    plot_confusion_matrix(baseline_model, base_val_loader, cfg, tag="Baseline")

    # ── Phase 4: Augmented Classifier (real + MC-WGAN-GP synthetic) ───
    print("\nPhase 4: Augmented Classifier (real + synthetic)")
    aug_train_loader, aug_val_loader = _make_clf_loaders(
        train_samples, val_samples, synth_paths, cfg
    )
    aug_model, augmented_acc = train_classifier(
        aug_train_loader, aug_val_loader, cfg, tag="MC-WGAN-GP Augmented"
    )
    plot_confusion_matrix(aug_model, aug_val_loader, cfg, tag="MC-WGAN-GP Augmented")

    # ── Phase 5: Compare ─────────────────────────────────────────────
    compare_results(baseline_acc, augmented_acc, cfg)

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Baseline Accuracy     : {baseline_acc:.2f}%")
    print(f"  MC-WGAN-GP Accuracy   : {augmented_acc:.2f}%")
    print(f"  Improvement           : {augmented_acc - baseline_acc:+.2f}%")
    print(f"  All outputs         → {cfg.OUTPUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()