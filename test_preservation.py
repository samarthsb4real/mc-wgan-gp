"""
Preservation Property Tests for MC-WGAN-GP Conditioning Bypass Fix

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8**

These tests observe and lock in the behavior of components that must remain
UNCHANGED after the fix is applied. They are written on UNFIXED code and
EXPECTED TO PASS — confirming the baseline behavior to preserve.

Components tested:
  1. MaskEncoder — deterministic, consistent encoding (Req 3.1)
  2. Label embedding — consistent for all 6 classes (Req 3.1)
  3. Stratified split — reproducible with same seed (Req 3.7)
  4. Gradient clipping — max norm = 1.0 after clip (Req 3.4)
  5. Checkpoint format — required keys exist (Req 3.5)
  6. Visualization grid — _sample_fixed_masks returns 24 masks (Req 3.8)
"""

import os
import sys
import math
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

sys.path.insert(0, os.path.dirname(__file__))

from run import (
    Config,
    MaskEncoder,
    MaskConditionedGenerator,
    MaskConditionedCritic,
    _load_all_samples,
    get_stratified_split,
    _sample_fixed_masks,
    save_checkpoint,
    set_seed,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

set_seed(42)
cfg = Config()
cfg.DEVICE = torch.device("cpu")

ALL_SAMPLES = _load_all_samples(cfg.DATA_DIR, cfg.CLASS_NAMES)
TRAIN_IDX, VAL_IDX = get_stratified_split(ALL_SAMPLES, seed=42)
TRAIN_SAMPLES = [ALL_SAMPLES[i] for i in TRAIN_IDX]


def make_generator():
    gen = MaskConditionedGenerator(
        cfg.LATENT_DIM, cfg.MASK_LATENT_DIM,
        cfg.NUM_CLASSES, cfg.GEN_FEATURES,
        cfg.CHANNELS, cfg.IMAGE_SIZE,
    ).to(cfg.DEVICE)
    set_seed(0)
    return gen


def make_critic():
    return MaskConditionedCritic(
        cfg.NUM_CLASSES, cfg.CRITIC_FEATURES, cfg.IMAGE_SIZE,
    ).to(cfg.DEVICE)


# ===========================================================================
# TEST 1: MaskEncoder Preservation
# **Validates: Requirements 3.1**
# ===========================================================================

class TestMaskEncoderPreservation:
    """
    MaskEncoder must produce deterministic, consistent outputs.
    Same input → same output (L2 distance < 1e-6).
    """

    def test_mask_encoder_deterministic(self):
        """Same mask input always produces the same encoding."""
        encoder = MaskEncoder(out_dim=128).to(cfg.DEVICE)
        encoder.eval()

        mask = torch.rand(1, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)
        mask = (mask > 0.5).float().to(cfg.DEVICE)

        with torch.no_grad():
            out1 = encoder(mask)
            out2 = encoder(mask)

        l2 = torch.dist(out1, out2).item()
        assert l2 < 1e-6, f"MaskEncoder is not deterministic: L2={l2}"

    def test_mask_encoder_output_shape(self):
        """MaskEncoder output shape is (B, 128)."""
        encoder = MaskEncoder(out_dim=128).to(cfg.DEVICE)
        encoder.eval()

        for batch_size in [1, 4, 8]:
            mask = torch.rand(batch_size, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(cfg.DEVICE)
            with torch.no_grad():
                out = encoder(mask)
            assert out.shape == (batch_size, 128), \
                f"Expected ({batch_size}, 128), got {out.shape}"

    def test_mask_encoder_100_random_masks_consistent(self):
        """
        Property: Generate 100 random masks, encode each twice, assert L2 < 1e-6.
        **Validates: Requirements 3.1**
        """
        encoder = MaskEncoder(out_dim=128).to(cfg.DEVICE)
        encoder.eval()

        set_seed(42)
        for i in range(100):
            mask = (torch.rand(1, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE) > 0.5).float().to(cfg.DEVICE)
            with torch.no_grad():
                out1 = encoder(mask)
                out2 = encoder(mask)
            l2 = torch.dist(out1, out2).item()
            assert l2 < 1e-6, f"Mask {i}: MaskEncoder non-deterministic, L2={l2}"

    def test_mask_encoder_different_masks_produce_different_outputs(self):
        """Different masks should produce different encodings (encoder is not constant)."""
        encoder = MaskEncoder(out_dim=128).to(cfg.DEVICE)
        encoder.eval()

        all_zeros = torch.zeros(1, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(cfg.DEVICE)
        all_ones  = torch.ones(1, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE).to(cfg.DEVICE)

        with torch.no_grad():
            enc_zeros = encoder(all_zeros)
            enc_ones  = encoder(all_ones)

        l2 = torch.dist(enc_zeros, enc_ones).item()
        assert l2 > 1e-3, f"MaskEncoder produces same output for all-zeros and all-ones masks: L2={l2}"


# ===========================================================================
# TEST 2: Label Embedding Preservation
# **Validates: Requirements 3.1**
# ===========================================================================

class TestLabelEmbeddingPreservation:
    """
    Label embedding must produce consistent outputs for all 6 class labels.
    Same label → same embedding (deterministic).
    """

    def test_label_embedding_all_6_classes_deterministic(self):
        """
        For all 6 class labels, embedding is deterministic (same input → same output).
        **Validates: Requirements 3.1**
        """
        gen = make_generator()
        gen.eval()

        for label_idx in range(cfg.NUM_CLASSES):
            label_tensor = torch.tensor([label_idx], device=cfg.DEVICE)
            with torch.no_grad():
                emb1 = gen.label_emb(label_tensor)
                emb2 = gen.label_emb(label_tensor)
            l2 = torch.dist(emb1, emb2).item()
            assert l2 < 1e-6, \
                f"Label {label_idx} ({cfg.DISPLAY_NAMES[label_idx]}): embedding non-deterministic, L2={l2}"

    def test_label_embedding_output_shape(self):
        """Label embedding output shape is (B, LATENT_DIM)."""
        gen = make_generator()
        gen.eval()

        for label_idx in range(cfg.NUM_CLASSES):
            label_tensor = torch.tensor([label_idx], device=cfg.DEVICE)
            with torch.no_grad():
                emb = gen.label_emb(label_tensor)
            assert emb.shape == (1, cfg.LATENT_DIM), \
                f"Label {label_idx}: expected (1, {cfg.LATENT_DIM}), got {emb.shape}"

    def test_label_embedding_different_classes_produce_different_embeddings(self):
        """Different class labels should produce different embeddings."""
        gen = make_generator()
        gen.eval()

        embeddings = {}
        for label_idx in range(cfg.NUM_CLASSES):
            label_tensor = torch.tensor([label_idx], device=cfg.DEVICE)
            with torch.no_grad():
                embeddings[label_idx] = gen.label_emb(label_tensor)

        for i in range(cfg.NUM_CLASSES):
            for j in range(i + 1, cfg.NUM_CLASSES):
                l2 = torch.dist(embeddings[i], embeddings[j]).item()
                assert l2 > 1e-3, \
                    f"Labels {i} and {j} produce identical embeddings: L2={l2}"


# ===========================================================================
# TEST 3: Stratified Split Preservation
# **Validates: Requirements 3.7**
# ===========================================================================

class TestStratifiedSplitPreservation:
    """
    Stratified split must be reproducible: same seed → same indices.
    """

    def test_stratified_split_reproducible_same_seed(self):
        """
        Same seed produces identical train/val indices.
        **Validates: Requirements 3.7**
        """
        train1, val1 = get_stratified_split(ALL_SAMPLES, seed=42)
        train2, val2 = get_stratified_split(ALL_SAMPLES, seed=42)

        assert train1 == train2, "Train indices differ with same seed"
        assert val1 == val2, "Val indices differ with same seed"

    def test_stratified_split_different_seeds_produce_different_splits(self):
        """Different seeds should produce different splits."""
        train1, val1 = get_stratified_split(ALL_SAMPLES, seed=42)
        train2, val2 = get_stratified_split(ALL_SAMPLES, seed=99)

        # At least some indices should differ
        assert set(train1) != set(train2) or set(val1) != set(val2), \
            "Different seeds produced identical splits"

    def test_stratified_split_10_seeds_all_reproducible(self):
        """
        Run stratified split 10 times with different seeds, verify each is reproducible.
        **Validates: Requirements 3.7**
        """
        seeds = [0, 1, 7, 13, 42, 99, 123, 256, 512, 1000]
        for seed in seeds:
            train1, val1 = get_stratified_split(ALL_SAMPLES, seed=seed)
            train2, val2 = get_stratified_split(ALL_SAMPLES, seed=seed)
            assert train1 == train2, f"Seed {seed}: train indices not reproducible"
            assert val1 == val2, f"Seed {seed}: val indices not reproducible"

    def test_stratified_split_preserves_class_proportions(self):
        """
        Stratified split preserves class proportions in train and val sets.
        **Validates: Requirements 3.7**
        """
        from collections import Counter

        train_idx, val_idx = get_stratified_split(ALL_SAMPLES, seed=42)
        train_samples = [ALL_SAMPLES[i] for i in train_idx]
        val_samples   = [ALL_SAMPLES[i] for i in val_idx]

        total_counts = Counter(s[2] for s in ALL_SAMPLES)
        train_counts = Counter(s[2] for s in train_samples)
        val_counts   = Counter(s[2] for s in val_samples)

        for label in range(cfg.NUM_CLASSES):
            total = total_counts.get(label, 0)
            if total == 0:
                continue
            train_ratio = train_counts.get(label, 0) / total
            val_ratio   = val_counts.get(label, 0) / total
            # Train should be ~85%, val ~15% (test_size=0.15)
            assert 0.70 <= train_ratio <= 0.95, \
                f"Class {label}: train ratio {train_ratio:.2f} out of expected range [0.70, 0.95]"
            assert 0.05 <= val_ratio <= 0.30, \
                f"Class {label}: val ratio {val_ratio:.2f} out of expected range [0.05, 0.30]"

    def test_stratified_split_no_overlap(self):
        """Train and val indices must not overlap."""
        train_idx, val_idx = get_stratified_split(ALL_SAMPLES, seed=42)
        overlap = set(train_idx) & set(val_idx)
        assert len(overlap) == 0, f"Train/val overlap: {len(overlap)} shared indices"

    def test_stratified_split_covers_all_samples(self):
        """Train + val indices must cover all samples."""
        train_idx, val_idx = get_stratified_split(ALL_SAMPLES, seed=42)
        all_idx = set(train_idx) | set(val_idx)
        assert len(all_idx) == len(ALL_SAMPLES), \
            f"Split covers {len(all_idx)} samples, expected {len(ALL_SAMPLES)}"


# ===========================================================================
# TEST 4: Gradient Clipping Preservation
# **Validates: Requirements 3.4**
# ===========================================================================

class TestGradientClippingPreservation:
    """
    Generator gradient clipping must clip at magnitude 1.0 (cfg.GRAD_CLIP).
    After clipping, max gradient norm must be <= 1.0.
    """

    def test_gradient_clip_value_is_1_0(self):
        """Config specifies GRAD_CLIP = 1.0."""
        assert cfg.GRAD_CLIP == 1.0, f"Expected GRAD_CLIP=1.0, got {cfg.GRAD_CLIP}"

    def test_gradient_clipping_max_norm_after_clip(self):
        """
        After gradient clipping, max parameter gradient norm <= 1.0.
        **Validates: Requirements 3.4**
        """
        gen = make_generator()
        gen.train()
        opt = optim.Adam(gen.parameters(), lr=cfg.LR_GEN)

        # Forward pass with random inputs
        B = 4
        noise  = torch.randn(B, cfg.LATENT_DIM)
        labels = torch.randint(0, cfg.NUM_CLASSES, (B,))
        masks  = (torch.rand(B, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE) > 0.5).float()

        fake, _pred_masks = gen(noise, labels, masks)
        loss = fake.mean()

        opt.zero_grad()
        loss.backward()

        # Record norms BEFORE clipping
        norms_before = [
            p.grad.norm().item()
            for p in gen.parameters()
            if p.grad is not None
        ]

        # Apply clipping
        nn.utils.clip_grad_norm_(gen.parameters(), cfg.GRAD_CLIP)

        # Record norms AFTER clipping
        norms_after = [
            p.grad.norm().item()
            for p in gen.parameters()
            if p.grad is not None
        ]

        max_norm_after = max(norms_after) if norms_after else 0.0
        print(f"\nGradient norms before clip: max={max(norms_before):.4f}, mean={np.mean(norms_before):.4f}")
        print(f"Gradient norms after  clip: max={max_norm_after:.4f}, mean={np.mean(norms_after):.4f}")

        assert max_norm_after <= cfg.GRAD_CLIP + 1e-5, \
            f"Max gradient norm after clipping = {max_norm_after:.4f} > {cfg.GRAD_CLIP}"

    def test_gradient_clipping_actually_clips_large_gradients(self):
        """
        When gradients are large, clipping reduces them.
        **Validates: Requirements 3.4**
        """
        gen = make_generator()
        gen.train()
        opt = optim.Adam(gen.parameters(), lr=cfg.LR_GEN)

        B = 4
        noise  = torch.randn(B, cfg.LATENT_DIM) * 100  # Large inputs to create large gradients
        labels = torch.randint(0, cfg.NUM_CLASSES, (B,))
        masks  = torch.ones(B, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)

        fake, _pred_masks = gen(noise, labels, masks)
        loss = (fake * 1000).mean()  # Scale up loss to create large gradients

        opt.zero_grad()
        loss.backward()

        norms_before = [
            p.grad.norm().item()
            for p in gen.parameters()
            if p.grad is not None
        ]
        max_before = max(norms_before) if norms_before else 0.0

        nn.utils.clip_grad_norm_(gen.parameters(), cfg.GRAD_CLIP)

        norms_after = [
            p.grad.norm().item()
            for p in gen.parameters()
            if p.grad is not None
        ]
        max_after = max(norms_after) if norms_after else 0.0

        # If gradients were large before, they should be smaller after
        if max_before > cfg.GRAD_CLIP:
            assert max_after <= cfg.GRAD_CLIP + 1e-5, \
                f"Clipping failed: before={max_before:.4f}, after={max_after:.4f}"
            print(f"\nClipping reduced max norm: {max_before:.4f} → {max_after:.4f}")


# ===========================================================================
# TEST 5: Checkpoint Format Preservation
# **Validates: Requirements 3.5**
# ===========================================================================

class TestCheckpointFormatPreservation:
    """
    Checkpoint must contain required keys: epoch, generator, critic, opt_gen, opt_critic.
    """

    REQUIRED_KEYS = ["epoch", "generator", "critic", "opt_gen", "opt_critic"]

    def test_checkpoint_contains_required_keys(self):
        """
        Save checkpoint and verify all required keys exist.
        **Validates: Requirements 3.5**
        """
        gen    = make_generator()
        critic = make_critic()
        opt_gen    = optim.Adam(gen.parameters(),    lr=cfg.LR_GEN)
        opt_critic = optim.Adam(critic.parameters(), lr=cfg.LR_CRITIC)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Temporarily override checkpoint dir
            original_dir = cfg.CHECKPOINT_DIR
            cfg.CHECKPOINT_DIR = tmpdir

            save_checkpoint(gen, critic, opt_gen, opt_critic, epoch=1, cfg=cfg)

            ckpt_path = os.path.join(tmpdir, "checkpoint_epoch_0001.pth")
            assert os.path.exists(ckpt_path), f"Checkpoint file not created: {ckpt_path}"

            ckpt = torch.load(ckpt_path, map_location="cpu")

            cfg.CHECKPOINT_DIR = original_dir

        for key in self.REQUIRED_KEYS:
            assert key in ckpt, f"Required key '{key}' missing from checkpoint"

    def test_checkpoint_epoch_value_correct(self):
        """Checkpoint epoch value matches the epoch passed to save_checkpoint."""
        gen    = make_generator()
        critic = make_critic()
        opt_gen    = optim.Adam(gen.parameters(),    lr=cfg.LR_GEN)
        opt_critic = optim.Adam(critic.parameters(), lr=cfg.LR_CRITIC)

        for epoch in [1, 10, 50, 100]:
            with tempfile.TemporaryDirectory() as tmpdir:
                original_dir = cfg.CHECKPOINT_DIR
                cfg.CHECKPOINT_DIR = tmpdir

                save_checkpoint(gen, critic, opt_gen, opt_critic, epoch=epoch, cfg=cfg)

                ckpt_path = os.path.join(tmpdir, f"checkpoint_epoch_{epoch:04d}.pth")
                ckpt = torch.load(ckpt_path, map_location="cpu")

                cfg.CHECKPOINT_DIR = original_dir

            assert ckpt["epoch"] == epoch, \
                f"Checkpoint epoch mismatch: expected {epoch}, got {ckpt['epoch']}"

    def test_checkpoint_generator_state_loadable(self):
        """Generator state dict saved in checkpoint can be loaded back."""
        gen    = make_generator()
        critic = make_critic()
        opt_gen    = optim.Adam(gen.parameters(),    lr=cfg.LR_GEN)
        opt_critic = optim.Adam(critic.parameters(), lr=cfg.LR_CRITIC)

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = cfg.CHECKPOINT_DIR
            cfg.CHECKPOINT_DIR = tmpdir

            save_checkpoint(gen, critic, opt_gen, opt_critic, epoch=5, cfg=cfg)

            ckpt_path = os.path.join(tmpdir, "checkpoint_epoch_0005.pth")
            ckpt = torch.load(ckpt_path, map_location="cpu")

            cfg.CHECKPOINT_DIR = original_dir

        # Load into a fresh generator
        gen2 = make_generator()
        gen2.load_state_dict(ckpt["generator"])

        # Verify weights are identical
        for (n1, p1), (n2, p2) in zip(gen.named_parameters(), gen2.named_parameters()):
            assert torch.allclose(p1, p2), f"Parameter {n1} differs after checkpoint load"

    def test_checkpoint_no_extra_required_keys_missing(self):
        """
        Checkpoint keys are exactly the required set (no missing keys).
        **Validates: Requirements 3.5**
        """
        gen    = make_generator()
        critic = make_critic()
        opt_gen    = optim.Adam(gen.parameters(),    lr=cfg.LR_GEN)
        opt_critic = optim.Adam(critic.parameters(), lr=cfg.LR_CRITIC)

        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = cfg.CHECKPOINT_DIR
            cfg.CHECKPOINT_DIR = tmpdir

            save_checkpoint(gen, critic, opt_gen, opt_critic, epoch=1, cfg=cfg)

            ckpt_path = os.path.join(tmpdir, "checkpoint_epoch_0001.pth")
            ckpt = torch.load(ckpt_path, map_location="cpu")

            cfg.CHECKPOINT_DIR = original_dir

        missing = [k for k in self.REQUIRED_KEYS if k not in ckpt]
        assert len(missing) == 0, f"Missing required checkpoint keys: {missing}"


# ===========================================================================
# TEST 6: Visualization Grid Preservation
# **Validates: Requirements 3.8**
# ===========================================================================

class TestVisualizationGridPreservation:
    """
    _sample_fixed_masks must return exactly 4 rows × 6 classes = 24 masks.
    Grid layout: 4 repetitions of one mask per class.
    """

    def test_sample_fixed_masks_returns_24_masks(self):
        """
        _sample_fixed_masks returns 4 rows × 6 classes = 24 masks.
        **Validates: Requirements 3.8**
        """
        masks = _sample_fixed_masks(TRAIN_SAMPLES, cfg)
        expected_count = 4 * cfg.NUM_CLASSES  # 4 rows × 6 classes = 24
        assert masks.shape[0] == expected_count, \
            f"Expected {expected_count} masks, got {masks.shape[0]}"

    def test_sample_fixed_masks_shape(self):
        """Each mask has shape (1, IMAGE_SIZE, IMAGE_SIZE)."""
        masks = _sample_fixed_masks(TRAIN_SAMPLES, cfg)
        assert masks.shape == (24, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE), \
            f"Expected (24, 1, {cfg.IMAGE_SIZE}, {cfg.IMAGE_SIZE}), got {masks.shape}"

    def test_sample_fixed_masks_4_rows_same_per_class(self):
        """
        Each class mask is repeated 4 times (one per row).
        Row 0 and row 1 for the same class should be identical.
        **Validates: Requirements 3.8**
        """
        masks = _sample_fixed_masks(TRAIN_SAMPLES, cfg)
        # masks layout: [row0_class0, row0_class1, ..., row0_class5,
        #                row1_class0, row1_class1, ..., row1_class5, ...]
        for cls in range(cfg.NUM_CLASSES):
            for row in range(1, 4):
                idx_row0 = cls                          # row 0, class cls
                idx_rowN = row * cfg.NUM_CLASSES + cls  # row N, class cls
                l2 = torch.dist(masks[idx_row0], masks[idx_rowN]).item()
                assert l2 < 1e-6, \
                    f"Class {cls}: mask at row 0 and row {row} differ (L2={l2:.6f})"

    def test_sample_fixed_masks_binary_values(self):
        """Masks contain only binary values (0.0 or 1.0)."""
        masks = _sample_fixed_masks(TRAIN_SAMPLES, cfg)
        unique_vals = masks.unique()
        for v in unique_vals:
            assert v.item() in [0.0, 1.0], \
                f"Non-binary value in masks: {v.item()}"

    def test_sample_fixed_masks_deterministic(self):
        """
        _sample_fixed_masks is deterministic for the same train_samples.
        **Validates: Requirements 3.8**
        """
        masks1 = _sample_fixed_masks(TRAIN_SAMPLES, cfg)
        masks2 = _sample_fixed_masks(TRAIN_SAMPLES, cfg)
        l2 = torch.dist(masks1.float(), masks2.float()).item()
        assert l2 < 1e-6, f"_sample_fixed_masks is not deterministic: L2={l2}"


# ===========================================================================
# PROPERTY-BASED TESTS (Hypothesis)
# **Validates: Requirements 3.1, 3.4, 3.7**
# ===========================================================================

class TestPropertyBased:
    """
    Property-based tests using Hypothesis for stronger guarantees.
    """

    @given(mask_seed=st.integers(min_value=0, max_value=99999))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    def test_mask_encoder_deterministic_property(self, mask_seed):
        """
        Property: For any binary mask, MaskEncoder produces identical output on two calls.
        **Validates: Requirements 3.1**
        """
        encoder = MaskEncoder(out_dim=128).to(cfg.DEVICE)
        encoder.eval()

        rng = np.random.RandomState(mask_seed)
        arr = (rng.rand(1, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE) > 0.5).astype(np.float32)
        mask = torch.from_numpy(arr).to(cfg.DEVICE)

        with torch.no_grad():
            out1 = encoder(mask)
            out2 = encoder(mask)

        l2 = torch.dist(out1, out2).item()
        assert l2 < 1e-6, f"MaskEncoder non-deterministic for given mask: L2={l2}"

    @given(seed=st.integers(min_value=0, max_value=9999))
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow])
    def test_stratified_split_reproducibility_property(self, seed):
        """
        Property: For any random seed, stratified split is reproducible.
        **Validates: Requirements 3.7**
        """
        train1, val1 = get_stratified_split(ALL_SAMPLES, seed=seed)
        train2, val2 = get_stratified_split(ALL_SAMPLES, seed=seed)
        assert train1 == train2, f"Seed {seed}: train indices not reproducible"
        assert val1 == val2, f"Seed {seed}: val indices not reproducible"

    @given(label=st.integers(min_value=0, max_value=5))
    @settings(max_examples=6, suppress_health_check=[HealthCheck.too_slow])
    def test_label_embedding_deterministic_property(self, label):
        """
        Property: For any class label in [0, 5], label embedding is deterministic.
        **Validates: Requirements 3.1**
        """
        gen = make_generator()
        gen.eval()

        label_tensor = torch.tensor([label], device=cfg.DEVICE)
        with torch.no_grad():
            emb1 = gen.label_emb(label_tensor)
            emb2 = gen.label_emb(label_tensor)

        l2 = torch.dist(emb1, emb2).item()
        assert l2 < 1e-6, f"Label {label}: embedding non-deterministic, L2={l2}"

    @given(
        batch_size=st.integers(min_value=1, max_value=8),
        scale=st.floats(min_value=1.0, max_value=100.0),
    )
    @settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much], deadline=None)
    def test_gradient_clipping_property(self, batch_size, scale):
        """
        Property: After gradient clipping with max_norm=1.0, all parameter
        gradient norms are <= 1.0.
        **Validates: Requirements 3.4**
        """
        gen = make_generator()
        gen.train()
        opt = optim.Adam(gen.parameters(), lr=cfg.LR_GEN)

        noise  = torch.randn(batch_size, cfg.LATENT_DIM) * scale
        labels = torch.zeros(batch_size, dtype=torch.long)
        masks  = torch.ones(batch_size, 1, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)

        fake, _pred_masks = gen(noise, labels, masks)
        loss = fake.mean() * scale

        opt.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(gen.parameters(), cfg.GRAD_CLIP)

        for name, param in gen.named_parameters():
            if param.grad is not None:
                norm = param.grad.norm().item()
                assert norm <= cfg.GRAD_CLIP + 1e-4, \
                    f"Param {name}: grad norm {norm:.4f} > GRAD_CLIP={cfg.GRAD_CLIP} after clipping"


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PRESERVATION PROPERTY TESTS — MC-WGAN-GP Conditioning Bypass")
    print("=" * 70)
    print("NOTE: These tests are EXPECTED TO PASS on unfixed code.")
    print("      They lock in baseline behavior to preserve after the fix.\n")
    pytest.main([__file__, "-v", "--tb=short", "-s"])
