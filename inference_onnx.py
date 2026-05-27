"""Inferensi teks dari gambar menggunakan model ONNX + ONNXRuntime CPU.

Usage:
  # Single image
  python inference_onnx.py --image_path path/to/receipt.jpg

  # Folder gambar
  python inference_onnx.py --image_dir path/to/images/

  # Custom model path
  python inference_onnx.py --image_path img.jpg --rec_model inference/rec_onnx/model.onnx
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_REC_MODEL = "./models/rec_model.onnx"
DICT_PATH = "./utils/en_dict.txt"
REC_IMAGE_SHAPE = (3, 48, 320)  # C, H, W — harus sesuai config training


# ── Label decoder ─────────────────────────────────────────────────────────────
class CTCDecoder:
    def __init__(self, dict_path: str, use_space_char: bool = True):
        chars = ["blank"]  # index 0 = CTC blank
        with open(dict_path, encoding="utf-8") as f:
            chars += [c.rstrip("\n") for c in f]
        if use_space_char:
            chars.append(" ")
        self.chars = chars

    def decode(self, pred: np.ndarray) -> str:
        """Greedy CTC decoding. pred: (T, num_classes)"""
        idx = np.argmax(pred, axis=-1)  # (T,)
        # Remove duplicates and blanks
        result = []
        prev = -1
        for i in idx:
            if i != prev and i != 0:
                result.append(self.chars[i] if i < len(self.chars) else "")
            prev = i
        return "".join(result)


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(img: np.ndarray, target_h: int = 48, target_w: int = 320) -> np.ndarray:
    """Resize keeping aspect ratio, pad width, normalize to [-1, 1]."""
    h, w = img.shape[:2]
    ratio = target_h / h
    new_w = min(int(w * ratio), target_w)
    img = cv2.resize(img, (new_w, target_h))
    # Pad to target_w
    pad_w = target_w - new_w
    if pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=127)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # normalize ke [-1, 1]
    img = img.transpose(2, 0, 1)  # HWC → CHW
    return img[np.newaxis, ...]  # add batch dim → (1, C, H, W)


# ── Inference ─────────────────────────────────────────────────────────────────
def load_session(model_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = os.cpu_count()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    return session


def predict(session: ort.InferenceSession, img_path: str, decoder: CTCDecoder) -> str:
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {img_path}")
    inp = preprocess(img)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: inp})
    # output[0]: (1, T, num_classes)  atau  (T, num_classes) — tergantung model
    pred = output[0]
    if pred.ndim == 3:
        pred = pred[0]  # ambil batch pertama
    return decoder.decode(pred)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PaddleOCR v4 ONNX Inference (CPU)")
    parser.add_argument("--rec_model", default=DEFAULT_REC_MODEL,
                        help="Path ke file .onnx hasil konversi")
    parser.add_argument("--dict_path", default=DICT_PATH,
                        help="Path ke character dictionary")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Path ke single gambar")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Path ke folder gambar (batch inference)")
    args = parser.parse_args()

    if not args.image_path and not args.image_dir:
        parser.error("Berikan --image_path atau --image_dir")

    print(f"[INFO] Loading ONNX model: {args.rec_model}")
    session = load_session(args.rec_model)
    decoder = CTCDecoder(args.dict_path)
    print(f"[INFO] Providers: {session.get_providers()}")

    if args.image_path:
        images = [args.image_path]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [str(p) for p in Path(args.image_dir).iterdir() if p.suffix.lower() in exts]
        images.sort()

    print(f"[INFO] Inferensi {len(images)} gambar...\n")
    for img_path in images:
        try:
            text = predict(session, img_path, decoder)
            print(f"{img_path}\t→  {text}")
        except FileNotFoundError as e:
            print(f"[WARN] {e}")


if __name__ == "__main__":
    main()
