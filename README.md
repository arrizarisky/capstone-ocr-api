# PaddleOCR v4 ONNX - Structured Receipt OCR API

Repository mandiri siap pakai untuk melakukan OCR struk/invoice secara end-to-end menggunakan model **PP-OCRv4** ONNX (DBNet++ untuk Detection dan SVTR-LCNet untuk Recognition) yang telah difine-tune. Service ini dibungkus menggunakan **FastAPI** untuk memudahkan integrasi via REST API.

## Fitur Utama
* **End-to-End Inference**: Deteksi posisi teks, warping perspektif wilayah teks, pembacaan tulisan (OCR), dan parsing data terstruktur dalam satu pipeline.
* **Fine-Tuned Model**: Menggunakan model recognition yang sudah di-optimize untuk mengenali struk belanja/invoice dengan akurasi tinggi.
* **REST API**: Endpoint `/ocr/receipt` yang menerima multipart upload gambar dan mengembalikan JSON terstruktur sesuai format spec AI-DS-SPEC.
* **Singleton Startup**: Model dimuat sekali di memori saat API server dimulai, memastikan respon yang sangat cepat.

---

## Struktur Folder
```
paddleocr-onnx-api/
├── models/
│   ├── det_model.onnx       # PP-OCRv4 Detection model
│   └── rec_model.onnx       # Fine-tuned PP-OCRv4 Recognition model (SVTR-LCNet)
├── utils/
│   └── en_dict.txt          # Kamus karakter PaddleOCR
├── test_images/             # Gambar contoh untuk pengujian
├── det_onnx.py              # Script preprocessing & postprocessing deteksi teks (DBNet++)
├── inference_onnx.py        # Script inferensi model recognition (CTC decoding)
├── parse_receipt.py         # Parser heuristik untuk memformat teks mentah jadi JSON terstruktur
├── ocr_receipt.py           # Pipeline orkestrator end-to-end
├── api.py                   # FastAPI service layer
├── test_api.py              # Automated test helper script
└── requirements.txt         # Daftar dependency package
```

---

## Cara Instalasi

1. **Pastikan Python 3.8+ sudah terinstall.**
2. **Install dependency**:
   ```bash
   pip install -r requirements.txt
   ```
   *(Rekomendasi: gunakan Virtual Environment)*

---

## Cara Menjalankan API Server

Jalankan FastAPI service dengan uvicorn:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Setelah server berjalan, Anda dapat mengakses dokumentasi interaktif Swagger UI di:
* http://localhost:8000/docs
* http://localhost:8000/redoc

---

## Contoh API Request & Response

### Request (cURL)
```bash
curl -X POST http://localhost:8000/ocr/receipt \
  -F "file=@test_images/receipt_00000.png"
```

### Response JSON (AI-DS-SPEC Fitur 2)
```json
{
  "merchant": "Unknown",
  "date": null,
  "total": 1591600,
  "items": [
    {
      "name": "NasiCampurBali",
      "qty": 1,
      "price": 125000
    },
    {
      "name": "OrganicGreen Sa",
      "qty": 1,
      "price": 65000
    }
  ],
  "category": "makanan",
  "confidence": 0.55
}
```

---

## Menjalankan Automated Test Script

Untuk mempermudah pengujian lokal, jalankan script pembantu:
```bash
python test_api.py
```
Script ini akan:
1. Memulai FastAPI server di background.
2. Mengirim data file gambar dari `test_images/` ke API endpoint.
3. Mencetak response JSON yang didapat dari server ke konsol.
4. Mematikan server background secara otomatis.
