"""
Pipeline end-to-end: gambar struk FULL → OCR → structured JSON (Fitur 2).

Pipeline:
  1. [DET] PP-OCRv4 det model (DBNet++) → deteksi text region
  2. [CROP] Perspective warp tiap region
  3. [REC] Model fine-tuned kamu (ONNX) → recognition teks
  4. [PARSE] parse_receipt.py → structured JSON

Usage:
  python ocr_receipt.py --image_path path/to/struk.jpg
  python ocr_receipt.py --image_path struk.jpg --output_json result.json
  python ocr_receipt.py --image_path struk.jpg --debug   # tampilkan visualisasi boxes

Output JSON (AI-DS-SPEC Fitur 2):
  {
    "merchant": "Indomaret",
    "date": "2026-05-27",
    "total": 15000,
    "items": ["Indomie Goreng", "Teh Botol x1"],
    "category": "makanan",
    "confidence": 0.85
  }
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from det_onnx import DetectionModel, crop_text_region
from inference_onnx import load_session, preprocess, CTCDecoder
from parse_receipt import parse_receipt


# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_DET_MODEL = "./models/det_model.onnx"
DEFAULT_REC_MODEL = "./models/rec_model.onnx"
DICT_PATH         = "./utils/en_dict.txt"


def run_det_rec(image_path: str,
               det_model: DetectionModel,
               rec_session,
               decoder: CTCDecoder,
               debug: bool = False) -> str:
    """
    Full pipeline: image → det boxes → crop → rec → sorted raw text.

    Args:
        image_path : path ke gambar input
        det_model  : DetectionModel instance
        rec_session: ONNXRuntime InferenceSession (rec model)
        decoder    : CTCDecoder instance
        debug      : jika True, simpan gambar dengan visualisasi boxes

    Returns:
        raw_text (str) — semua teks terdeteksi, dipisah newline per baris
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    # ── Step 1: Detection ──────────────────────────────────────────────────────
    print(f"[DET] Mendeteksi text regions...")
    boxes = det_model.detect(img)
    print(f"[DET] Ditemukan {len(boxes)} text region")

    if len(boxes) == 0:
        print("[WARN] Tidak ada text region terdeteksi. Coba turunkan DB_BOX_THRESH di det_onnx.py")
        return ""

    # ── Debug visualization ────────────────────────────────────────────────────
    if debug:
        debug_img = img.copy()
        for i, box in enumerate(boxes):
            cv2.polylines(debug_img, [box], True, (0, 255, 0), 2)
            cv2.putText(debug_img, str(i), tuple(box[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        debug_path = Path(image_path).stem + "_debug.jpg"
        cv2.imwrite(debug_path, debug_img)
        print(f"[DEBUG] Visualisasi boxes disimpan ke: {debug_path}")

    # ── Step 2 & 3: Crop + Recognize ──────────────────────────────────────────
    input_name = rec_session.get_inputs()[0].name
    lines = []

    for i, box in enumerate(boxes):
        # Crop & warp perspective region
        crop = crop_text_region(img, box)
        if crop is None or crop.size == 0:
            continue

        # Preprocess untuk rec model (resize ke [3, 48, 320])
        inp = preprocess(crop, target_h=48)
        output = rec_session.run(None, {input_name: inp})
        pred = output[0]
        if pred.ndim == 3:
            pred = pred[0]

        text = decoder.decode(pred).strip()
        if text:
            lines.append(text)
            print(f"  [{i:03d}] {text}")

    raw_text = "\n".join(lines)
    return raw_text


def process_receipt(image_path: str,
                   det_model_path: str = DEFAULT_DET_MODEL,
                   rec_model_path: str = DEFAULT_REC_MODEL,
                   dict_path: str = DICT_PATH,
                   debug: bool = False) -> dict:
    """End-to-end: gambar full → JSON sesuai AI-DS-SPEC Fitur 2."""

    # Load models
    print(f"[INFO] Loading DET model: {det_model_path}")
    det = DetectionModel(det_model_path)

    print(f"[INFO] Loading REC model: {rec_model_path}")
    rec_session = load_session(rec_model_path)
    decoder = CTCDecoder(dict_path)

    # Run pipeline
    raw_text = run_det_rec(image_path, det, rec_session, decoder, debug=debug)

    print(f"\n[REC] Raw OCR output:\n{'='*40}")
    print(raw_text)
    print("="*40)

    # Parse ke structured JSON
    print("[INFO] Parsing structured fields...")
    result = parse_receipt(raw_text)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="OCR Struk/Invoice Full Image → JSON (AI-DS-SPEC Fitur 2)"
    )
    parser.add_argument("--image_path",   required=True,              help="Path ke gambar struk (JPG/PNG)")
    parser.add_argument("--det_model",    default=DEFAULT_DET_MODEL,  help="Path ke det model .onnx")
    parser.add_argument("--rec_model",    default=DEFAULT_REC_MODEL,  help="Path ke rec model .onnx")
    parser.add_argument("--dict_path",    default=DICT_PATH,          help="Path ke character dict")
    parser.add_argument("--output_json",  default=None,               help="Simpan hasil ke file JSON")
    parser.add_argument("--debug",        action="store_true",         help="Simpan visualisasi detection boxes")
    args = parser.parse_args()

    result = process_receipt(
        args.image_path,
        det_model_path=args.det_model,
        rec_model_path=args.rec_model,
        dict_path=args.dict_path,
        debug=args.debug,
    )

    print("\n[RESULT] Structured JSON:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[OK] Saved to: {args.output_json}")


if __name__ == "__main__":
    main()
