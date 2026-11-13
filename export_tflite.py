"""
export_tflite.py - Standalone TensorFlow Lite Export Module
ForensiX-DeepFake (Cross-Framework Portability Engine)

Exports the trained Dual-Architecture Combo (EfficientNet-B0 + ConvNeXt-Tiny)
from PyTorch weights into optimized TensorFlow Lite FlatBuffers:
  - model_combo.tflite (Float32 full precision)
  - model_combo_fp16.tflite (Float16 compact precision)

Usage:
    python export_tflite.py --weights ./model_weights.pt --output_dir ./
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models


def build_effnet_b0(dropout=0.4):
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(m.classifier[1].in_features, 1))
    return m


def build_convnext():
    m = models.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
    return m


class DualCombo(nn.Module):
    """Dual-Architecture Ensemble Combo: averages raw logits of B0 and ConvNeXt, returns sigmoid probability."""
    def __init__(self, m1, m2):
        super().__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, x):
        z1 = self.m1(x).clamp(-8.0, 8.0)
        z2 = self.m2(x).clamp(-8.0, 8.0)
        avg_logit = (z1 + z2) / 2.0
        return torch.sigmoid(avg_logit)


def load_member_state(checkpoint, key_variants):
    if isinstance(checkpoint, dict) and "models" in checkpoint:
        for mem in checkpoint["models"]:
            if any(k.lower() == mem["name"].lower() for k in key_variants):
                return mem["state_dict"]
    for k in key_variants:
        if k in checkpoint:
            return checkpoint[k]
    return None


def main():
    parser = argparse.ArgumentParser(description="Export Dual-Architecture Combo to TensorFlow Lite")
    parser.add_argument("--weights", default="model_weights.pt", help="Path to model_weights.pt")
    parser.add_argument("--output_dir", default=".", help="Directory to save exported .tflite files")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version (default: 14)")
    args = parser.parse_args()

    weights_path = Path(args.weights).resolve()
    if not weights_path.exists():
        sys.exit(f"[ERROR] Weights file not found: {weights_path}")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"[1/4] Loading trained PyTorch weights from: {weights_path.name}")
    print("=" * 65)

    checkpoint = torch.load(weights_path, map_location="cpu")
    b0_dict = load_member_state(checkpoint, ["b0_corrupt", "b0_base", "efficientnet_b0"])
    cn_dict = load_member_state(checkpoint, ["convnext", "convnext_tiny"])

    if b0_dict is None or cn_dict is None:
        sys.exit("[ERROR] Could not find EfficientNet-B0 and ConvNeXt-Tiny weights in checkpoint.")

    b0 = build_effnet_b0()
    b0.load_state_dict(b0_dict, strict=True)
    b0.eval()

    cn = build_convnext()
    cn.load_state_dict(cn_dict, strict=True)
    cn.eval()
    print("  Successfully initialized EfficientNet-B0 and ConvNeXt-Tiny backbones.")

    combo_pt = DualCombo(b0, cn).eval()

    try:
        import onnx
        import onnx2tf
        import tensorflow as tf
    except ImportError as e:
        print(f"[WARN] Optional packages for TFLite compilation not installed in current env: {e}")
        print("  To compile TFLite on your local machine, run:")
        print("    pip install onnx onnx2tf tensorflow")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        combo_onnx_path = os.path.join(tmpdir, "model_combo.onnx")
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

        print("\n" + "=" * 65)
        print(f"[2/4] Tracing and exporting ONNX graph (Opset {args.opset})...")
        print("=" * 65)

        torch.onnx.export(
            combo_pt,
            dummy_input,
            combo_onnx_path,
            input_names=["input"],
            output_names=["probability"],
            dynamic_axes={"input": {0: "batch_size"}, "probability": {0: "batch_size"}},
            opset_version=args.opset,
            dynamo=False
        )

        print("\n" + "=" * 65)
        print("[3/4] Converting ONNX -> TensorFlow SavedModel via onnx2tf...")
        print("=" * 65)
        tf_saved_model_dir = os.path.join(tmpdir, "tf_model")
        onnx2tf.convert(
            input_onnx_file_path=combo_onnx_path,
            output_folder_path=tf_saved_model_dir,
            non_verbose=True
        )

        print("\n" + "=" * 65)
        print("[4/4] Compiling TensorFlow Lite FlatBuffers (FP32 & FP16)...")
        print("=" * 65)

        # FP32
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_fp32 = converter.convert()
        fp32_path = out_dir / "model_combo.tflite"
        with open(fp32_path, "wb") as f:
            f.write(tflite_fp32)
        print(f"  [SUCCESS] Wrote Float32 FlatBuffer: {fp32_path} ({len(tflite_fp32) / (1024*1024):.1f} MB)")

        # FP16
        converter_fp16 = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_dir)
        converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_fp16.target_spec.supported_types = [tf.float16]
        tflite_fp16 = converter_fp16.convert()
        fp16_path = out_dir / "model_combo_fp16.tflite"
        with open(fp16_path, "wb") as f:
            f.write(tflite_fp16)
        print(f"  [SUCCESS] Wrote Float16 FlatBuffer: {fp16_path} ({len(tflite_fp16) / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    main()
