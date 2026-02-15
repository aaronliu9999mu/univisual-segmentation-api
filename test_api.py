"""
Smoke test for the Univisual Segmentation API.
Run with: python test_api.py

Expects the server to already be running on http://localhost:8000
"""

import sys
import requests
import time

BASE = "http://localhost:8000"
PASS = 0
FAIL = 0


def check(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}  — {detail}")


def wait_for_server(timeout: int = 60):
    """Wait for the server to be ready."""
    print(f"⏳ Waiting for server at {BASE} (up to {timeout}s)...")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{BASE}/health", timeout=3)
            if r.status_code == 200:
                print("  Server is up!\n")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    print("  ⚠️  Server did not start in time.\n")
    return False


def test_root():
    print("── GET / ──")
    r = requests.get(BASE, timeout=10)
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    data = r.json()
    check("has 'message' key", "message" in data)


def test_health():
    print("── GET /health ──")
    r = requests.get(f"{BASE}/health", timeout=10)
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    data = r.json()
    check("status is healthy", data.get("status") == "healthy", f"got {data.get('status')}")
    models = data.get("models", {})
    check("cellpose is loaded", models.get("cellpose") is True, f"cellpose={models.get('cellpose')}")


def test_models():
    print("── GET /models ──")
    r = requests.get(f"{BASE}/models", timeout=10)
    check("status 200", r.status_code == 200, f"got {r.status_code}")
    data = r.json()
    model_list = data.get("models", [])
    check("has models list", len(model_list) > 0, "empty list")
    cellpose_models = [m for m in model_list if m.get("id") == "cellpose"]
    check("cellpose model listed", len(cellpose_models) == 1)
    if cellpose_models:
        check("cellpose available", cellpose_models[0].get("available") is True)


def test_segment_no_file():
    """Verify the API returns a proper error when no file is sent."""
    print("── POST /segment (no file) ──")
    r = requests.post(f"{BASE}/segment", timeout=10)
    check("returns 422 (validation error)", r.status_code == 422, f"got {r.status_code}")


def test_segment_with_synthetic_image():
    """Send a tiny synthetic PNG to /segment and verify it processes."""
    print("── POST /segment (synthetic 64x64 image) ──")
    try:
        from PIL import Image
        import io
        import numpy as np

        # Create a small synthetic image with some bright spots (fake nuclei)
        img_arr = np.zeros((64, 64, 3), dtype=np.uint8)
        # Add some bright circles
        for cx, cy in [(16, 16), (48, 48), (16, 48), (48, 16)]:
            for dx in range(-4, 5):
                for dy in range(-4, 5):
                    if dx*dx + dy*dy <= 16:
                        img_arr[cy+dy, cx+dx] = [200, 200, 255]

        img = Image.fromarray(img_arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        files = {"file": ("test.png", buf, "image/png")}
        data = {"model": "cellpose"}
        r = requests.post(f"{BASE}/segment", files=files, data=data, timeout=120)
        check("status 200", r.status_code == 200, f"got {r.status_code}")
        if r.status_code == 200:
            result = r.json()
            check("has 'success' key", result.get("success") is True)
            check("has 'cell_count' key", "cell_count" in result)
            check("has 'cells' list", isinstance(result.get("cells"), list))
            print(f"    → Detected {result.get('cell_count', '?')} cells")
    except ImportError:
        print("  ⚠️  Skipped (PIL/numpy not available)")


if __name__ == "__main__":
    if not wait_for_server():
        print("Cannot connect to server. Start it first with: python main.py")
        sys.exit(1)

    test_root()
    test_health()
    test_models()
    test_segment_no_file()
    test_segment_with_synthetic_image()

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL > 0:
        print("Some tests FAILED!")
        sys.exit(1)
    else:
        print("All tests PASSED! ✅")
        sys.exit(0)
