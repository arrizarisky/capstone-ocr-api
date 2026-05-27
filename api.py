"""FastAPI wrapper untuk OCR Struk/Invoice (AI-DS-SPEC Fitur 2).
Exposes the end-to-end PP-OCRv4 + Fine-tuned ONNX pipeline.

Usage:
  pip install fastapi uvicorn python-multipart opencv-python-headless shapely onnxruntime
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from det_onnx import DetectionModel
from inference_onnx import CTCDecoder, load_session
from ocr_receipt import run_det_rec
from parse_receipt import parse_receipt

# ── Config ────────────────────────────────────────────────────────────────────
DET_MODEL_PATH = "./models/det_model.onnx"
REC_MODEL_PATH = "./models/rec_model.onnx"
DICT_PATH = "./utils/en_dict.txt"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILE_SIZE_MB = 10

# ── Schemas ───────────────────────────────────────────────────────────────────
class ReceiptItem(BaseModel):
    name: str = Field(..., description="Nama item/barang")
    qty: int = Field(1, description="Kuantitas barang")
    price: int = Field(..., description="Harga total baris untuk item tersebut")

class ReceiptResponse(BaseModel):
    merchant: str = Field(..., description="Nama merchant/toko")
    date: Optional[str] = Field(None, description="Tanggal transaksi format YYYY-MM-DD atau null")
    total: Optional[int] = Field(None, description="Total nominal belanja")
    items: List[Union[ReceiptItem, str]] = Field(default_factory=list, description="Daftar item belanja")
    category: str = Field("lainnya", description="Kategori transaksi (makanan, transport, belanja, dll)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Tingkat kepercayaan parser (0.0 - 1.0)")

# ── App Init ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="OCR Receipt ONNX API",
    description="FastAPI service untuk OCR struk belanja menggunakan pipeline PP-OCRv4 ONNX (DBNet++ & SVTR-LCNet).",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
det_model = None
rec_session = None
decoder = None

@app.on_event("startup")
async def load_models():
    global det_model, rec_session, decoder
    print("[API] Initializing models...")
    
    if not os.path.exists(DET_MODEL_PATH):
        raise FileNotFoundError(f"Detection model not found at {DET_MODEL_PATH}")
    if not os.path.exists(REC_MODEL_PATH):
        raise FileNotFoundError(f"Recognition model not found at {REC_MODEL_PATH}")
    if not os.path.exists(DICT_PATH):
        raise FileNotFoundError(f"Dictionary file not found at {DICT_PATH}")
        
    print(f"[API] Loading Detection model from: {DET_MODEL_PATH}")
    det_model = DetectionModel(DET_MODEL_PATH)
    
    print(f"[API] Loading Recognition model from: {REC_MODEL_PATH}")
    rec_session = load_session(REC_MODEL_PATH)
    
    print(f"[API] Loading dictionary from: {DICT_PATH}")
    decoder = CTCDecoder(DICT_PATH)
    
    print("[API] Initialization complete. Service ready.")

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "OCR Receipt ONNX API is running.",
        "documentation": "/docs",
        "endpoints": {
            "/health": "GET - Check status",
            "/ocr/receipt": "POST - Upload image & extract structured receipt JSON"
        }
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models": {
            "detection_loaded": det_model is not None,
            "recognition_loaded": rec_session is not None,
            "decoder_loaded": decoder is not None
        }
    }

@app.post("/ocr/receipt", response_model=ReceiptResponse)
async def extract_receipt(file: UploadFile = File(...)):
    """Upload gambar struk/invoice, ekstrak teks dengan OCR, dan kembalikan JSON terstruktur."""
    # 1. Validasi file extension
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format file tidak didukung: {suffix}. Gunakan salah satu dari: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # 2. Baca file dan validasi ukuran
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Ukuran file melebihi batas maksimal ({MAX_FILE_SIZE_MB}MB)"
        )

    # 3. Validasi kevalidan gambar
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=422,
            detail="File corrupt atau tidak dapat di-decode sebagai gambar."
        )

    # 4. Simpan ke temporary file karena run_det_rec membutuhkan file path
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # 5. Jalankan OCR Pipeline
        print(f"[API] Processing image: {file.filename}")
        raw_text = run_det_rec(
            image_path=tmp_path,
            det_model=det_model,
            rec_session=rec_session,
            decoder=decoder,
            debug=False
        )
        
        # 6. Parse teks mentah menjadi JSON terstruktur
        result = parse_receipt(raw_text)
        
        print(f"[API] Successfully processed: {file.filename} -> Merchant: {result.get('merchant')}")
        return result
        
    except Exception as e:
        print(f"[API] Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gagal melakukan OCR: {str(e)}")
        
    finally:
        # Selalu bersihkan file temp
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
