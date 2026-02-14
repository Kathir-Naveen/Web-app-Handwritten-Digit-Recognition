"""
Digit Recognition API — FastAPI Backend
Serves both the prediction API and the frontend as static files.
Model: digit_recognition_model.h5 (CNN, 99.56% accuracy on MNIST)
"""

import os
import time
import base64
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("digit-api")

# ─── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    str(Path(__file__).parent / "digit_recognition_model.h5"),
)
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB
FRONTEND_DIR = str(Path(__file__).parent.parent / "frontend")

# ─── Global state ───────────────────────────────────────────────────────────
model: tf.keras.Model | None = None
stats = {"predictions": 0, "started_at": None}


# ─── Preprocessing pipeline ────────────────────────────────────────────────
def preprocess_user_drawing(image_data: str) -> np.ndarray:
    """
    Convert a base64-encoded canvas drawing to a (1, 28, 28, 1) tensor
    matching MNIST format.  All 10 steps from PROJECT_SPEC.md are included.

    Steps:
      1. Decode base64 → raw bytes → grayscale image
      2. Check for empty canvas
      3. Invert if white background (MNIST = black bg, white digit)
      4. Binary threshold (Otsu) to remove anti-aliasing
      5. Find bounding box of digit
      6. Crop with padding
      7. Resize to 20×20 maintaining aspect ratio
      8. Center in 28×28 canvas
      9. Normalize to [0, 1]
     10. Reshape to (1, 28, 28, 1)
    """
    try:
        # Step 1 — decode
        if "base64," in image_data:
            image_data = image_data.split("base64,")[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Failed to decode image data")

        # Step 2 — empty canvas
        if np.sum(img) == 0:
            return np.zeros((1, 28, 28, 1), dtype=np.float32)

        # Step 3 — invert if white background
        if np.mean(img) > 127:
            img = 255 - img

        # Step 4 — binary threshold (Otsu)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Step 5 — find bounding box
        coords = cv2.findNonZero(img)
        if coords is None:
            return np.zeros((1, 28, 28, 1), dtype=np.float32)

        x, y, w, h = cv2.boundingRect(coords)

        # Step 6 — crop with padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img.shape[1] - x, w + 2 * padding)
        h = min(img.shape[0] - y, h + 2 * padding)
        cropped = img[y : y + h, x : x + w]

        # Step 7 — resize to 20×20 maintaining aspect ratio
        if w > h:
            new_w = 20
            new_h = max(1, int(20 * h / w))
        else:
            new_h = 20
            new_w = max(1, int(20 * w / h))

        new_w = max(1, min(20, new_w))
        new_h = max(1, min(20, new_h))
        resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Step 8 — center in 28×28
        final_img = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - new_w) // 2
        y_offset = (28 - new_h) // 2
        final_img[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

        # Step 9 — normalize
        final_img = final_img.astype("float32") / 255.0

        # Step 10 — reshape
        return final_img.reshape(1, 28, 28, 1)

    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise ValueError(f"Image preprocessing failed: {e}")


# ─── Lifespan (model loading) ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info(f"Loading model from {MODEL_PATH} …")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    logger.info("✅ Model loaded")

    # Warm-up inference to pre-compile the graph
    dummy = np.zeros((1, 28, 28, 1), dtype=np.float32)
    _ = model.predict(dummy, verbose=0)
    logger.info("✅ Warm-up inference complete")

    stats["started_at"] = datetime.now().isoformat()
    yield
    model = None
    logger.info("Model unloaded — shutdown complete")


# ─── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Digit Recognition API",
    description="Handwritten digit recognition using a CNN trained on MNIST",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded PNG from canvas.toDataURL()")

    @field_validator("image")
    @classmethod
    def validate_image(cls, v: str) -> str:
        if not v:
            raise ValueError("Image data is required")
        if len(v) > MAX_IMAGE_SIZE:
            raise ValueError("Image exceeds 5 MB limit")
        return v


class PredictionResponse(BaseModel):
    digit: int
    confidence: float
    all_probabilities: Dict[str, float]
    inference_time_ms: float


# ─── API Routes ─────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "predictions": stats["predictions"],
        "uptime_since": stats["started_at"],
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        t0 = time.perf_counter()

        processed = preprocess_user_drawing(request.image)

        # Empty canvas guard
        if np.sum(processed) == 0:
            raise HTTPException(status_code=400, detail="Empty canvas — draw something first!")

        predictions = model.predict(processed, verbose=0)[0]

        digit = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        all_probs = {str(i): round(float(predictions[i]), 6) for i in range(10)}
        elapsed_ms = (time.perf_counter() - t0) * 1000

        stats["predictions"] += 1
        logger.info(f"Prediction #{stats['predictions']}: {digit}  ({confidence:.1%})  {elapsed_ms:.1f} ms")

        return PredictionResponse(
            digit=digit,
            confidence=confidence,
            all_probabilities=all_probs,
            inference_time_ms=round(elapsed_ms, 2),
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed — please try again")


# ─── Serve frontend ────────────────────────────────────────────────────────
@app.get("/")
async def serve_frontend():
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"message": "Digit Recognition API", "docs": "/docs"}


# Mount static files (CSS, JS, images if any)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
