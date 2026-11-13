"""
tf_predict.py - Standalone Zero-PyTorch TensorFlow Lite Inference Pipeline
ForensiX-DeepFake Engine

Runs high-performance inference entirely via TensorFlow Lite FlatBuffer & NumPy.
Works offline with zero PyTorch dependency.

Usage:
    python tf_predict.py --test_dir ./test/images/ --model ./model_combo.tflite --output predictions_tflite.csv
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        sys.exit("[ERROR] tensorflow or tflite_runtime is required: pip install tensorflow")

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
EXTS = {".jpg", ".jpeg", ".png"}


def preprocess(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = (np.array(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return arr  # (224, 224, 3) for channels-last TF or transposed for channels-first


def main():
    parser = argparse.ArgumentParser(description="Standalone TensorFlow Lite Inference")
    parser.add_argument("--test_dir", required=True, help="Folder containing unlabelled test images")
    parser.add_argument("--model", default="model_combo.tflite", help="Path to .tflite FlatBuffer")
    parser.add_argument("--output", default="predictions_tflite.csv", help="Path to output CSV")
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    test_folder = Path(args.test_dir)
    paths = sorted([p for p in test_folder.iterdir() if p.suffix.lower() in EXTS])
    if not paths:
        sys.exit(f"[ERROR] No valid images found in {test_folder}")

    print(f"[INFO] Initializing TensorFlow Lite Interpreter: {args.model}")
    interpreter = Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    is_channels_first = (input_shape[1] == 3)

    print(f"[INFO] Model input shape: {input_shape} ({'NCHW' if is_channels_first else 'NHWC'})")
    print(f"[INFO] Processing {len(paths)} images...")
    t0 = time.time()
    ids, probs = [], []

    for p in paths:
        arr = preprocess(p)
        if is_channels_first:
            arr = arr.transpose(2, 0, 1)
        inp = np.expand_dims(arr, axis=0).astype(np.float32)

        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"]).squeeze()

        # Sigmoid if raw logit
        if output < 0.0 or output > 1.0:
            prob = 1.0 / (1.0 + np.exp(-output))
        else:
            prob = float(output)

        probs.append(prob)
        ids.append(p.stem)

    probs = np.clip(np.array(probs), 0.0, 1.0)
    labels = (probs >= args.threshold).astype(int)

    df = pd.DataFrame({"image_id": ids, "label": labels, "confidence": probs})
    df.to_csv(args.output, index=False)
    print(f"[SUCCESS] Exported {len(df)} predictions to {args.output} in {time.time() - t0:.2f}s")
    print(f"  Predicted FAKE (1): {labels.sum()} ({labels.mean()*100:.1f}%) | REAL (0): {(labels==0).sum()}")


if __name__ == "__main__":
    main()
