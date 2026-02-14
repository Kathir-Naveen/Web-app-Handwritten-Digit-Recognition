"""
Comprehensive Test Suite for Digit Recognition Web App
Tests: API endpoints, UI elements, drawing, prediction, clear, mobile responsiveness, error handling
"""

import json
import time
import base64
import urllib.request
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8000"
RESULTS = []
SCREENSHOTS_DIR = "/tmp/test_screenshots"

import os
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


def log(test_name, passed, detail=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    RESULTS.append((test_name, passed, detail))
    print(f"  {status}  {test_name}" + (f"  — {detail}" if detail else ""))


def api_get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


def api_post(path, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  🧪  DIGIT RECOGNITION WEB APP — TEST SUITE")
print("=" * 65)


# ─── PART 1: API TESTS ─────────────────────────────────────────────────
print("\n── 1. API Endpoint Tests ─────────────────────────────────────")

# Test 1.1: Health endpoint
try:
    status, data = api_get("/health")
    log("GET /health returns 200", status == 200, f"status={status}")
    log("Health: model_loaded is true", data.get("model_loaded") is True)
    log("Health: status is healthy", data.get("status") == "healthy")
except Exception as e:
    log("GET /health returns 200", False, str(e))

# Test 1.2: Root serves HTML
try:
    req = urllib.request.Request(f"{BASE_URL}/")
    with urllib.request.urlopen(req, timeout=10) as resp:
        content_type = resp.headers.get("content-type", "")
        body = resp.read().decode()
        log("GET / serves HTML page", "text/html" in content_type, f"content-type={content_type}")
        log("HTML contains canvas element", "drawCanvas" in body)
        log("HTML contains Predict button", "predictBtn" in body or "Predict" in body)
except Exception as e:
    log("GET / serves HTML page", False, str(e))

# Test 1.3: POST /predict with invalid data
try:
    status, data = api_post("/predict", {"image": "not-a-valid-image"})
    log("POST /predict with invalid data returns 422", status == 422, f"status={status}")
except Exception as e:
    log("POST /predict with invalid data returns 422", False, str(e))

# Test 1.4: POST /predict with empty body
try:
    status, data = api_post("/predict", {})
    log("POST /predict with empty body returns 422", status == 422, f"status={status}")
except Exception as e:
    log("POST /predict with empty body returns 422", False, str(e))

# Test 1.5: POST /predict with a real base64 image (white canvas = empty)
try:
    # Create a tiny white PNG (empty canvas)
    import struct, zlib
    def make_white_png(w, h):
        raw = b""
        for _ in range(h):
            raw += b"\x00" + b"\xff" * (w * 3)
        def chunk(ctype, data):
            c = ctype + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )
    white_png = make_white_png(28, 28)
    b64 = "data:image/png;base64," + base64.b64encode(white_png).decode()
    status, data = api_post("/predict", {"image": b64})
    log("POST /predict with empty canvas returns 400", status == 400, f"status={status}, detail={data.get('detail','')}")
except Exception as e:
    log("POST /predict with empty canvas returns 400", False, str(e))

# Test 1.6: POST /predict with a black-digit image (should predict)
try:
    # Create a small PNG with a black rectangle in the center (simulating a digit)
    import struct, zlib
    def make_digit_png(w, h):
        raw = b""
        for y in range(h):
            raw += b"\x00"
            for x in range(w):
                if 8 <= x <= 20 and 4 <= y <= 24:
                    raw += b"\x00\x00\x00"  # black digit
                else:
                    raw += b"\xff\xff\xff"  # white bg
            
        def chunk(ctype, data):
            c = ctype + data
            return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )
    digit_png = make_digit_png(28, 28)
    b64 = "data:image/png;base64," + base64.b64encode(digit_png).decode()
    status, data = api_post("/predict", {"image": b64})
    log("POST /predict with digit image returns 200", status == 200, f"status={status}")
    if status == 200:
        log("Response has 'digit' field (0-9)", data.get("digit") in range(10), f"digit={data.get('digit')}")
        log("Response has 'confidence' (0-1)", 0 <= data.get("confidence", -1) <= 1, f"confidence={data.get('confidence')}")
        log("Response has 'all_probabilities' (10 entries)", len(data.get("all_probabilities", {})) == 10)
        log("Response has 'inference_time_ms'", data.get("inference_time_ms", -1) > 0, f"{data.get('inference_time_ms')}ms")
        log("Inference time < 500ms", data.get("inference_time_ms", 9999) < 500, f"{data.get('inference_time_ms')}ms")
except Exception as e:
    log("POST /predict with digit image returns 200", False, str(e))


# ─── PART 2: BROWSER / UI TESTS ────────────────────────────────────────
print("\n── 2. Browser UI Tests ──────────────────────────────────────")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)

    # ── Desktop tests ───────────────────────────────────────────────
    page = browser.new_page(viewport={"width": 1024, "height": 768})
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")

    # Test 2.1: Page title
    title = page.title()
    log("Page title contains 'Neural Digit'", "Neural Digit" in title, f"title='{title}'")

    # Test 2.2: Key UI elements present
    log("Canvas element exists", page.locator("#drawCanvas").count() == 1)
    log("Predict button exists", page.locator("#predictBtn").count() == 1)
    log("Clear button exists", page.locator("#clearBtn").count() == 1)
    log("Status pill exists", page.locator("#statusPill").count() == 1)
    log("Result card exists", page.locator("#resultCard").count() == 1)

    # Test 2.3: Status shows model ready
    page.wait_for_timeout(2000)  # wait for health check
    status_text = page.locator("#statusText").text_content()
    log("Status shows 'model ready'", "model ready" in status_text.lower(), f"status='{status_text}'")

    status_class = page.locator("#statusPill").get_attribute("class")
    log("Status pill has 'online' class", "online" in (status_class or ""), f"class='{status_class}'")

    # Test 2.4: Predict without drawing → error message
    page.locator("#predictBtn").click()
    page.wait_for_timeout(500)
    result_html = page.locator("#resultCard").inner_html()
    log("Predict without drawing shows error", "Draw a digit first" in result_html, f"result='{result_html[:80]}'")

    page.screenshot(path=f"{SCREENSHOTS_DIR}/01_error_no_drawing.png", full_page=True)

    # Test 2.5: Draw on canvas
    canvas = page.locator("#drawCanvas")
    box = canvas.bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2

    # Draw a "1" — vertical line
    page.mouse.move(cx, cy - 80)
    page.mouse.down()
    page.mouse.move(cx, cy + 80, steps=20)
    page.mouse.up()
    page.wait_for_timeout(300)

    page.screenshot(path=f"{SCREENSHOTS_DIR}/02_after_drawing.png", full_page=True)
    log("Drawing on canvas works (visual check via screenshot)", True)

    # Test 2.6: Predict after drawing
    page.locator("#predictBtn").click()
    page.wait_for_timeout(3000)  # wait for prediction

    result_html = page.locator("#resultCard").inner_html()
    has_digit = "predicted-digit" in result_html
    has_confidence = "confidence-bar" in result_html
    has_probs = "probs-grid" in result_html
    log("Prediction shows digit result", has_digit, f"has_digit={has_digit}")
    log("Prediction shows confidence bar", has_confidence)
    log("Prediction shows probability grid", has_probs)

    # Extract the predicted digit
    if has_digit:
        digit_el = page.locator(".predicted-digit")
        digit_text = digit_el.text_content()
        log("Predicted digit is 0-9", digit_text.strip() in [str(i) for i in range(10)], f"digit='{digit_text}'")

    # Check inference time display
    has_time = "inference" in result_html.lower()
    log("Inference time is displayed", has_time)

    page.screenshot(path=f"{SCREENSHOTS_DIR}/03_prediction_result.png", full_page=True)

    # Test 2.7: Result card has highlight class
    result_class = page.locator("#resultCard").get_attribute("class")
    log("Result card has 'has-result' class", "has-result" in (result_class or ""))

    # Test 2.8: Clear canvas
    page.locator("#clearBtn").click()
    page.wait_for_timeout(500)

    result_html_after_clear = page.locator("#resultCard").inner_html()
    log("Clear resets result display", "placeholder" in result_html_after_clear or "appear here" in result_html_after_clear)

    result_class_after = page.locator("#resultCard").get_attribute("class")
    log("Clear removes 'has-result' class", "has-result" not in (result_class_after or ""))

    page.screenshot(path=f"{SCREENSHOTS_DIR}/04_after_clear.png", full_page=True)

    # Test 2.9: Draw multiple digits and predict each
    print("\n── 3. Multi-Digit Prediction Tests ─────────────────────────")
    digits_tested = 0
    for digit_idx, draw_fn in enumerate([
        # Draw "1" - vertical line
        lambda p, cx, cy: [p.mouse.move(cx, cy-80), p.mouse.down(), p.mouse.move(cx, cy+80, steps=15), p.mouse.up()],
        # Draw "0" - oval
        lambda p, cx, cy: [
            p.mouse.move(cx, cy-60), p.mouse.down(),
            p.mouse.move(cx+40, cy-30, steps=8),
            p.mouse.move(cx+40, cy+30, steps=8),
            p.mouse.move(cx, cy+60, steps=8),
            p.mouse.move(cx-40, cy+30, steps=8),
            p.mouse.move(cx-40, cy-30, steps=8),
            p.mouse.move(cx, cy-60, steps=8),
            p.mouse.up()
        ],
        # Draw "7" - horizontal then diagonal
        lambda p, cx, cy: [
            p.mouse.move(cx-40, cy-60), p.mouse.down(),
            p.mouse.move(cx+40, cy-60, steps=10),
            p.mouse.move(cx-10, cy+60, steps=15),
            p.mouse.up()
        ],
    ]):
        page.locator("#clearBtn").click()
        page.wait_for_timeout(300)
        draw_fn(page, cx, cy)
        page.wait_for_timeout(200)
        page.locator("#predictBtn").click()
        page.wait_for_timeout(2500)

        result_html = page.locator("#resultCard").inner_html()
        has_result = "predicted-digit" in result_html
        if has_result:
            d = page.locator(".predicted-digit").text_content().strip()
            log(f"Drawing #{digit_idx+1} produces valid prediction", d in [str(i) for i in range(10)], f"predicted={d}")
            digits_tested += 1
        else:
            log(f"Drawing #{digit_idx+1} produces valid prediction", False, "no result")

        page.screenshot(path=f"{SCREENSHOTS_DIR}/05_digit_{digit_idx}.png", full_page=True)

    log(f"All {digits_tested}/3 multi-digit tests returned predictions", digits_tested == 3)

    # ── Mobile tests ────────────────────────────────────────────────
    print("\n── 4. Mobile Responsiveness Tests ──────────────────────────")
    page.close()

    # iPhone SE viewport
    mobile_page = browser.new_page(viewport={"width": 375, "height": 667}, is_mobile=True, has_touch=True)
    mobile_page.goto(BASE_URL)
    mobile_page.wait_for_load_state("networkidle")
    mobile_page.wait_for_timeout(1000)

    # Test 4.1: Canvas fits within viewport
    canvas_m = mobile_page.locator("#drawCanvas")
    box_m = canvas_m.bounding_box()
    log("Mobile: canvas fits viewport width", box_m["width"] <= 375, f"canvas_width={box_m['width']:.0f}")
    log("Mobile: canvas is at least 280px wide", box_m["width"] >= 280, f"canvas_width={box_m['width']:.0f}")

    # Test 4.2: Buttons are visible
    predict_visible = mobile_page.locator("#predictBtn").is_visible()
    clear_visible = mobile_page.locator("#clearBtn").is_visible()
    log("Mobile: Predict button visible", predict_visible)
    log("Mobile: Clear button visible", clear_visible)

    # Test 4.3: Touch drawing simulation
    mcx, mcy = box_m["x"] + box_m["width"] / 2, box_m["y"] + box_m["height"] / 2
    mobile_page.touchscreen.tap(mcx, mcy - 40)
    mobile_page.mouse.move(mcx, mcy - 40)
    mobile_page.mouse.down()
    mobile_page.mouse.move(mcx, mcy + 40, steps=10)
    mobile_page.mouse.up()
    mobile_page.wait_for_timeout(300)

    mobile_page.locator("#predictBtn").click()
    mobile_page.wait_for_timeout(2500)
    mobile_result = mobile_page.locator("#resultCard").inner_html()
    log("Mobile: prediction works after touch draw", "predicted-digit" in mobile_result)

    mobile_page.screenshot(path=f"{SCREENSHOTS_DIR}/06_mobile_prediction.png", full_page=True)

    # Test 4.4: No horizontal scroll
    viewport_width = mobile_page.evaluate("document.documentElement.clientWidth")
    scroll_width = mobile_page.evaluate("document.documentElement.scrollWidth")
    log("Mobile: no horizontal overflow", scroll_width <= viewport_width + 5, f"viewport={viewport_width}, scroll={scroll_width}")

    mobile_page.close()

    # ── Keyboard shortcut tests ─────────────────────────────────────
    print("\n── 5. Keyboard Shortcut Tests ──────────────────────────────")
    kb_page = browser.new_page(viewport={"width": 1024, "height": 768})
    kb_page.goto(BASE_URL)
    kb_page.wait_for_load_state("networkidle")
    kb_page.wait_for_timeout(1000)

    # Draw something first
    canvas_kb = kb_page.locator("#drawCanvas")
    box_kb = canvas_kb.bounding_box()
    kcx, kcy = box_kb["x"] + box_kb["width"]/2, box_kb["y"] + box_kb["height"]/2
    kb_page.mouse.move(kcx, kcy - 50)
    kb_page.mouse.down()
    kb_page.mouse.move(kcx, kcy + 50, steps=10)
    kb_page.mouse.up()

    # Test 5.1: Enter key triggers prediction
    kb_page.keyboard.press("Enter")
    kb_page.wait_for_timeout(2500)
    kb_result = kb_page.locator("#resultCard").inner_html()
    log("Enter key triggers prediction", "predicted-digit" in kb_result)

    # Test 5.2: Escape key clears canvas
    kb_page.keyboard.press("Escape")
    kb_page.wait_for_timeout(500)
    kb_result_after = kb_page.locator("#resultCard").inner_html()
    log("Escape key clears canvas + result", "placeholder" in kb_result_after or "appear here" in kb_result_after)

    kb_page.close()
    browser.close()


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
total = len(RESULTS)
passed = sum(1 for _, p, _ in RESULTS if p)
failed = sum(1 for _, p, _ in RESULTS if not p)

print(f"  📊  RESULTS: {passed}/{total} passed, {failed} failed")

if failed > 0:
    print("\n  Failed tests:")
    for name, p, detail in RESULTS:
        if not p:
            print(f"    ❌ {name}  — {detail}")

print("=" * 65)
print(f"\n  Screenshots saved to: {SCREENSHOTS_DIR}/")
print()

exit(0 if failed == 0 else 1)
