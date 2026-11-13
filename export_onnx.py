"""
export_onnx.py - Standalone ONNX Export and Numerical Parity Verification Module
ForensiX-DeepFake System

Exports trained PyTorch weights into self-contained ONNX computational graphs
with parameters embedded internally (zero external .data dependency).
Supports:
- Dual-Architecture Combo (EfficientNet-B0 + ConvNeXt-Tiny) [Default / Recommended]
- Pruned 4-Model Ensemble (B0 corrupt + B0 base + ConvNeXt + Swin)
- Full 5-Model Ensemble (B0 corrupt + B0 base + ConvNeXt + ViT + Swin)
- Single Backbones (b0_corrupt, convnext, etc.)

Usage:
    # 1. Export Dual-Architecture Combo (B0 + ConvNeXt):
    python export_onnx.py --weights ./model_weights.pt --output ./model_dual.onnx --model dual --verify

    # 2. Export Single Backbone (b0_corrupt):
    python export_onnx.py --weights ./model_weights.pt --output ./model.onnx --model b0_corrupt --verify

    # 3. Export Pruned 4-Model Ensemble:
    python export_onnx.py --weights ./model_weights.pt --output ./model_ensemble.onnx --model ensemble_pruned --verify
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torchvision import models


def build_effnet_b0(dropout: float = 0.4) -> nn.Module:
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(m.classifier[1].in_features, 1)
    )
    return m


def build_convnext() -> nn.Module:
    m = models.convnext_tiny(weights=None)
    m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
    return m


def build_effnet_b3(dropout: float = 0.3) -> nn.Module:
    m = models.efficientnet_b3(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(m.classifier[1].in_features, 1)
    )
    return m


def build_vit() -> nn.Module:
    m = models.vit_b_16(weights=None)
    m.heads.head = nn.Linear(m.heads.head.in_features, 1)
    return m


def build_swin() -> nn.Module:
    m = models.swin_t(weights=None)
    m.head = nn.Linear(m.head.in_features, 1)
    return m


class DualComboONNX(nn.Module):
    """Dual-Architecture Ensemble Combo: EfficientNet + ConvNeXt-Tiny."""
    def __init__(self, m1, m2):
        super().__init__()
        self.m1 = m1
        self.m2 = m2

    def forward(self, x):
        z1 = self.m1(x).clamp(-8.0, 8.0)
        z2 = self.m2(x).clamp(-8.0, 8.0)
        mean_logit = (z1 + z2) / 2.0
        return torch.sigmoid(mean_logit)


class EnsembleONNX(nn.Module):
    """Multi-Architecture Ensemble: averages raw logits and returns probability."""
    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, x):
        logits = [net(x).clamp(-8.0, 8.0) for net in self.nets]
        mean_logit = torch.stack(logits, dim=0).mean(dim=0)
        return torch.sigmoid(mean_logit)


def load_member_state(checkpoint, key_variants):
    """Finds matching state dict across different container structures."""
    if isinstance(checkpoint, dict) and "models" in checkpoint:
        for mem in checkpoint["models"]:
            if any(k.lower() == mem["name"].lower() for k in key_variants):
                return mem["state_dict"]
    for k in key_variants:
        if k in checkpoint:
            return checkpoint[k]
    return None


def export_model_to_onnx(weights_path: str, output_path: str, target_model: str = "dual", opset_version: int = 14):
    print(f"[INFO] Loading PyTorch weights from: {weights_path}")
    checkpoint = torch.load(weights_path, map_location="cpu")

    if target_model == "dual":
        # Dual combo: b0_corrupt (or b0_base) + convnext
        b0_state = load_member_state(checkpoint, ["b0_corrupt", "b0_base", "efficientnet_b0"])
        cn_state = load_member_state(checkpoint, ["convnext", "convnext_tiny"])

        if b0_state is None or cn_state is None:
            raise SystemExit("[ERROR] Checkpoint must contain both EfficientNet-B0 and ConvNeXt for 'dual' export.")

        b0 = build_effnet_b0()
        b0.load_state_dict(b0_state, strict=True)
        b0.eval()

        cn = build_convnext()
        cn.load_state_dict(cn_state, strict=True)
        cn.eval()

        model = DualComboONNX(b0, cn).eval()
        print("[INFO] Initialized Dual-Architecture Combo (EffNet-B0 + ConvNeXt-Tiny)")

    elif target_model in ["ensemble", "ensemble_pruned"]:
        all_specs = [
            (["b0_corrupt"], build_effnet_b0),
            (["b0_base"], build_effnet_b0),
            (["convnext", "convnext_tiny"], build_convnext),
            (["swin", "swin_t"], build_swin),
        ]
        if target_model == "ensemble":
            all_specs.append((["vit", "vit_b_16"], build_vit))

        nets = []
        for keys, builder in all_specs:
            state = load_member_state(checkpoint, keys)
            if state is not None:
                m = builder()
                m.load_state_dict(state, strict=True)
                m.eval()
                nets.append(m)
        if not nets:
            raise SystemExit("[ERROR] Could not find ensemble members in checkpoint.")
        model = EnsembleONNX(nets).eval()
        print(f"[INFO] Initialized Ensemble with {len(nets)} backbones")

    elif target_model in ["b0_corrupt", "b0_base", "efficientnet_b0"]:
        state = load_member_state(checkpoint, [target_model, "b0_corrupt", "b0_base"])
        model = build_effnet_b0()
        model.load_state_dict(state if state is not None else checkpoint, strict=False)
        model.eval()

    elif target_model in ["convnext", "convnext_tiny"]:
        state = load_member_state(checkpoint, ["convnext", "convnext_tiny"])
        model = build_convnext()
        model.load_state_dict(state if state is not None else checkpoint, strict=False)
        model.eval()

    else:
        raise SystemExit(f"[ERROR] Unsupported target_model: '{target_model}'")

    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    print(f"[INFO] Exporting target '{target_model}' to self-contained ONNX graph: {output_path} (Opset {opset_version})")
    export_kwargs = dict(
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["prob_fake"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "prob_fake": {0: "batch_size"}
        }
    )
    try:
        torch.onnx.export(model, dummy_input, output_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, dummy_input, output_path, **export_kwargs)

    # Ensure strictly self-contained (zero external .data dependency)
    data_sidecar = output_path + ".data"
    if os.path.exists(data_sidecar):
        try:
            import onnx
            m_onnx = onnx.load(output_path, load_external_data=True)
            onnx.save_model(m_onnx, output_path, save_as_external_data=False)
            if os.path.exists(data_sidecar):
                os.remove(data_sidecar)
        except ImportError:
            pass

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[SUCCESS] Successfully created {output_path} ({file_size_mb:.2f} MB)")
    return model, dummy_input


def verify_onnx(model, dummy_input, onnx_path: str):
    print("[INFO] Verifying ONNX structural integrity and runtime parity...")
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        print(f"[WARN] Skipping ONNX verification because optional dependency is missing: {exc}")
        return

    assert not os.path.exists(onnx_path + ".data"), "External .data file must not exist!"
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[OK] ONNX structural integrity check passed (100% self-contained).")

    with torch.no_grad():
        pt_out = model(dummy_input).squeeze().numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_inputs = {"input": dummy_input.numpy()}
    ort_out = sess.run(None, ort_inputs)[0].squeeze()

    max_diff = np.max(np.abs(pt_out - ort_out))
    print(f"[INFO] Maximum absolute error between PyTorch and ONNX: {max_diff:.6e}")
    assert max_diff < 1e-4, f"Parity verification failed: max_diff = {max_diff}"
    print("[OK] NUMERICAL PARITY VERIFIED: PyTorch and ONNX predictions match within 1e-4 tolerance.")


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model(s) to ONNX with numerical parity verification.")
    parser.add_argument("--weights", type=str, default="model_weights.pt", help="Path to model_weights.pt")
    parser.add_argument("--output", type=str, default="model_combo.onnx", help="Path to output model.onnx")
    parser.add_argument("--model", type=str, default="dual",
                        choices=["dual", "ensemble_pruned", "ensemble", "b0_corrupt", "b0_base", "convnext"],
                        help="Target architecture to export: 'dual' (B0+ConvNeXt), 'ensemble_pruned', 'ensemble', 'b0_corrupt', 'convnext'")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--verify", action="store_true", help="Run onnxruntime verification")
    args = parser.parse_args()

    model, dummy_input = export_model_to_onnx(args.weights, args.output, args.model, args.opset)
    if args.verify:
        verify_onnx(model, dummy_input, args.output)


if __name__ == "__main__":
    main()
