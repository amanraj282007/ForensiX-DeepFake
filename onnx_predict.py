"""
onnx_predict.py - Standalone Zero-PyTorch ONNX Runtime Inference Pipeline
ForensiX-DeepFake Engine

Runs high-performance batched inference entirely via ONNX Runtime & NumPy.
Works offline with zero PyTorch dependency.

Usage:
    python onnx_predict.py --test_dir ./test/images/ --model ./model.onnx --output predictions_onnx.csv
"""

import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    sys.exit("[ERROR] onnxruntime is required: pip install onnxruntime")

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
EXTS = {".jpg", ".jpeg", ".png"}


def preprocess(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = (np.array(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)  # (3, 224, 224)


def main():
    parser = argparse.ArgumentParser(description="Standalone ONNX Runtime Inference")
    parser.add_argument("--test_dir", required=True, help="Folder containing unlabelled test images")
    parser.add_argument("--model", default="model.onnx", help="Path to exported ONNX graph")
    parser.add_argument("--output", default="predictions_onnx.csv", help="Path to output CSV")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    test_folder = Path(args.test_dir)
    paths = sorted([p for p in test_folder.iterdir() if p.suffix.lower() in EXTS])
    if not paths:
        sys.exit(f"[ERROR] No valid images found in {test_folder}")

    print(f"[INFO] Initializing ONNX Runtime session: {args.model}")
    sess = ort.InferenceSession(args.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    print(f"[INFO] Processing {len(paths)} images with batch size {args.batch_size}...")
    t0 = time.time()
    ids, probs = [], []

    for i in range(0, len(paths), args.batch_size):
        batch_paths = paths[i:i + args.batch_size]
        batch_tensors = np.stack([preprocess(p) for p in batch_paths], axis=0)
        outputs = sess.run(None, {input_name: batch_tensors})[0].reshape(-1)

        # If model outputs raw logits rather than probabilities:
        if (outputs < 0).any() or (outputs > 1).any():
            outputs = 1.0 / (1.0 + np.exp(-outputs))

        probs.extend(outputs.tolist())
        ids.extend([p.stem for p in batch_paths])

    probs = np.clip(np.array(probs), 0.0, 1.0)
    labels = (probs >= args.threshold).astype(int)

    df = pd.DataFrame({"image_id": ids, "label": labels, "confidence": probs})
    df.to_csv(args.output, index=False)
    print(f"[SUCCESS] Exported {len(df)} predictions to {args.output} in {time.time() - t0:.2f}s")
    print(f"  Predicted FAKE (1): {labels.sum()} ({labels.mean()*100:.1f}%) | REAL (0): {(labels==0).sum()}")


if __name__ == "__main__":
    main()
