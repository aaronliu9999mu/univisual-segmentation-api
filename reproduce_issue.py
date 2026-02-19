import requests
import time
from PIL import Image
import numpy as np
import io
import os

# Assuming backend is on port 8000
BASE_URL = "http://localhost:8000/segment"

def create_large_image():
    """Generates a dummy large image."""
    print("Generating synthetic image (approx 2000x2000 uncompressed)...")
    # 2000x2000x3 bytes = 12MB raw data. 
    # TIFF might compress it, but we can disable compression or use noise to make it hard to compress.
    width, height = 2000, 2000
    # Random noise to prevent easy compression
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    
    # Save as uncompressed TIFF
    buf = io.BytesIO()
    img.save(buf, format="TIFF", compression=None)
    size_mb = len(buf.getvalue()) / 1024 / 1024
    print(f"Created image size: {size_mb:.2f} MB")
    buf.seek(0)
    return buf, img, size_mb

def test_endpoint(image_buffer, format_name):
    start = time.time()
    try:
        # Reset buffer pointer
        image_buffer.seek(0)
        files = {"file": (f"test.{format_name.lower()}", image_buffer, f"image/{format_name.lower()}")}
        data = {"model": "cellpose"}  # or whatever model is available
        
        print(f"  Sending {format_name} payload to {BASE_URL}...")
        response = requests.post(BASE_URL, files=files, data=data, timeout=60)
        
        duration = time.time() - start
        if response.status_code == 200:
            print(f"  ✅ Success! Time: {duration:.2f}s")
            return duration
        elif response.status_code == 422 or response.status_code == 400:
             print(f"  ⚠️  Server rejected (Validation): {response.status_code}")
             return duration # still counts as a round trip
        else:
            print(f"  ❌ Failed! Status: {response.status_code}, Time: {duration:.2f}s")
            print(f"     Error: {response.text[:200]}")
            return None
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Connection Error: Backend not running at {BASE_URL}")
        return None
    except requests.exceptions.ReadTimeout:
        print(f"  ❌ Timeout Error (>60s)")
        return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def run():
    print("--- Simulating Timeout Fix ---")
    
    # 1. Create large dummy image
    original_buf, img_obj, orig_size = create_large_image()
    
    # Check if backend is running first
    try:
        requests.get("http://localhost:8000/health", timeout=2)
    except:
        print("❌ Backend is not running on localhost:8000.")
        print("   Please start the backend with 'python main.py' to run this test.")
        return

    # 2. Test Original Upload
    print(f"\n[Test 1] Uploading Original {orig_size:.2f} MB TIFF (Simulating Timeout Cause)")
    t1 = test_endpoint(original_buf, "TIFF")
    
    # 3. Simulate Client-Side Optimization (Resize + JPEG)
    print("\n[Test 2] Simulating Client-Side Optimization (Resize + JPEG)")
    optimized_buf = io.BytesIO()
    
    # Resize logic similar to frontend
    max_dim = 2048
    w, h = img_obj.size
    if w > max_dim or h > max_dim:
        img_obj.thumbnail((max_dim, max_dim))
        
    img_obj.save(optimized_buf, format="JPEG", quality=80)
    opt_size = len(optimized_buf.getvalue()) / 1024 / 1024
    print(f"Optimized size: {opt_size:.2f} MB")
    
    t2 = test_endpoint(optimized_buf, "JPEG")
    
    if t1 is not None and t2 is not None:
        print(f"\n=== Result Summary ===")
        print(f"Original ({orig_size:.2f}MB):  {t1:.2f}s")
        print(f"Optimized ({opt_size:.2f}MB): {t2:.2f}s")
        diff = t1 - t2
        print(f"Time Saved:      {diff:.2f}s")
        print(f"Speedup Factor:  {t1/t2:.1f}x")
        
        if t2 < 10:
             print("✅ Optimized request is fast enough (<10s) to avoid timeouts!")
        else:
             print("⚠️ Optimized request is still > 10s.")

if __name__ == "__main__":
    run()
