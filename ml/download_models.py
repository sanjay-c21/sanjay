"""
Model Downloader & Verifier.
Downloads the pre-trained Microsoft FERPlus Emotion ONNX model
and verifies OpenCV Haar Cascade files.
"""

import os
import urllib.request
import cv2

MODEL_URL = "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx"
BACKUP_URL = "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
TARGET_DIR = "models"
MODEL_FILE = os.path.join(TARGET_DIR, "emotion-ferplus-8.onnx")

def download_file_with_progress(url, dest_path):
    print(f"Downloading model from: {url}")
    print(f"Destination: {dest_path}")
    
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
        else:
            mb_downloaded = downloaded / (1024 * 1024)
            print(f"\rDownloaded {mb_downloaded:.1f} MB...", end="", flush=True)
            
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    
    with urllib.request.urlopen(req, timeout=60) as response, open(dest_path, 'wb') as out_file:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 128  # 128 KB
        downloaded = 0
        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            downloaded += len(buffer)
            out_file.write(buffer)
            if total_size > 0:
                percent = min(100, downloaded * 100 / total_size)
                print(f"\rProgress: {percent:.1f}% ({downloaded/(1024*1024):.1f}/{total_size/(1024*1024):.1f} MB)", end="", flush=True)
    print("\nDownload complete!")

def ensure_models():
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # 1. Check/Download FERPlus ONNX model
    if os.path.exists(MODEL_FILE) and os.path.getsize(MODEL_FILE) > 30 * 1024 * 1024:
        print(f"FERPlus model already present ({os.path.getsize(MODEL_FILE)/(1024*1024):.1f} MB) at {MODEL_FILE}")
    else:
        try:
            download_file_with_progress(MODEL_URL, MODEL_FILE)
        except Exception as e:
            print(f"\nPrimary download failed ({e}). Trying backup URL...")
            download_file_with_progress(BACKUP_URL, MODEL_FILE)
            
    # Verify model with OpenCV DNN
    try:
        net = cv2.dnn.readNetFromONNX(MODEL_FILE)
        print("Successfully validated ONNX model with OpenCV DNN engine!")
    except Exception as e:
        print(f"Warning verifying ONNX with cv2.dnn: {e}")
        
    # 2. Check OpenCV Haar Cascade
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if os.path.exists(cascade_path):
        print(f"OpenCV Haar Cascade verified at: {cascade_path}")
    else:
        print(f"Haar Cascade not found at {cascade_path}. Verifying fallback.")

if __name__ == "__main__":
    ensure_models()
