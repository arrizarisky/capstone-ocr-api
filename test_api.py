"""Script helper untuk mengetes API secara lokal.
Menjalankan API server secara otomatis, menembak endpoint /ocr/receipt dengan file gambar test,
dan menampilkan response JSON terstruktur yang dihasilkan.

Usage:
  python test_api.py
"""

import subprocess
import time
import sys
import os
from pathlib import Path

# Cek dependencies
try:
    import requests
except ImportError:
    print("[ERROR] Library 'requests' dibutuhkan untuk menjalankan script testing.")
    print("        Silakan install via: pip install requests")
    sys.exit(1)

API_PORT = 8000
API_URL = f"http://localhost:{API_PORT}/ocr/receipt"
TEST_IMAGE = "./test_images/receipt_00000.png"

def start_server():
    print(f"[TEST] Memulai server API di port {API_PORT}...")
    # Menjalankan uvicorn di background
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--port", str(API_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return proc

def wait_for_server(proc):
    retries = 15
    while retries > 0:
        # Cek jika proses uvicorn mati sebelum startup selesai
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            print("[ERROR] Server gagal dijalankan!")
            print(f"Stdout:\n{stdout}")
            print(f"Stderr:\n{stderr}")
            sys.exit(1)
            
        try:
            r = requests.get(f"http://localhost:{API_PORT}/health")
            if r.status_code == 200:
                data = r.json()
                if data.get("models", {}).get("recognition_loaded") is True:
                    print("[TEST] Server aktif dan model berhasil dimuat!")
                    return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
        retries -= 1
        
    print("[ERROR] Timeout menunggu server aktif.")
    proc.terminate()
    sys.exit(1)

def test_ocr():
    if not os.path.exists(TEST_IMAGE):
        print(f"[ERROR] Gambar test tidak ditemukan di: {TEST_IMAGE}")
        sys.exit(1)
        
    print(f"[TEST] Mengirim gambar {TEST_IMAGE} ke endpoint API: {API_URL}...")
    start_time = time.time()
    
    with open(TEST_IMAGE, "rb") as f:
        files = {"file": (os.path.basename(TEST_IMAGE), f, "image/png")}
        response = requests.post(API_URL, files=files)
        
    duration = time.time() - start_time
    print(f"[TEST] Selesai dalam {duration:.2f} detik.")
    
    if response.status_code == 200:
        print("\n[TEST] RESPONSE SUCCESS:")
        import json
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"\n[TEST] RESPONSE ERROR (Code: {response.status_code}):")
        print(response.text)

def main():
    # Pindah ke directory project agar paths relatif bekerja dengan benar
    os.chdir(Path(__file__).parent.resolve())
    
    # Jalankan server
    proc = start_server()
    
    try:
        # Tunggu uvicorn running
        wait_for_server(proc)
        # Test request
        test_ocr()
    finally:
        # Shutdown server
        print("[TEST] Mematikan server API...")
        proc.terminate()
        proc.wait()
        print("[TEST] Server dimatikan.")

if __name__ == "__main__":
    main()
