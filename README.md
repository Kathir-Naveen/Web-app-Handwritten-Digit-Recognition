<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/FastAPI-0.109+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/TensorFlow-2.16+-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Render-Deployed-46E3B7?logo=render&logoColor=white" alt="Render">
  <img src="https://img.shields.io/badge/MNIST_Accuracy-99.56%25-brightgreen" alt="Accuracy">
  <br><br>
  <a href="https://handwritten-digit-recognition-yu98.onrender.com/"><img src="https://img.shields.io/badge/🔴_Live_Demo-Try_it_Now-7c3aed?style=for-the-badge" alt="Live Demo"></a>
</p>

# ✍️ Neural Digit — AI Handwritten Digit Recognition

> Draw a digit on a canvas and watch a Convolutional Neural Network predict it in real-time — with **99.56% accuracy** on MNIST.

A full-stack web application featuring a **sci-fi themed** drawing canvas, a **FastAPI** backend serving a pre-trained CNN model, and one-click deployment via **Docker** and **Render**.

---

## 🎥 Live Demo

🌐 **Try it now → [handwritten-digit-recognition-yu98.onrender.com](https://handwritten-digit-recognition-yu98.onrender.com/)**

| Draw | Predict | Result |
|------|---------|--------|
| ✏️ Sketch any digit (0–9) on the neon canvas | 🚀 Hit **Predict** or press `Enter` | 🔮 See the predicted digit, confidence score, and full probability distribution |

---

## ✨ Features

- **Real-time CNN Inference** — Pre-trained Keras model with 99.56% accuracy on MNIST
- **10-Step Image Preprocessing Pipeline** — Decode → Invert → Threshold → Crop → Resize → Center → Normalize
- **Interactive Dark/Light Canvas** — Sci-fi "Midnight Galaxy" theme with particle effects, nebula blobs, and neon glow
- **Confidence Visualization** — Animated bar + probability grid for all 10 digits
- **Keyboard Shortcuts** — `Enter` to predict, `Escape` to clear
- **Mobile Responsive** — Touch-drawing support and adaptive layout for all screen sizes
- **Health Monitoring** — `/health` endpoint with uptime, model status, and prediction count
- **Docker & Render Ready** — Multi-stage Dockerfile + `render.yaml` for instant cloud deployment

---

## 🏗️ Architecture

```
Hand_written_prediction_website/
├── backend/
│   ├── main.py                        # FastAPI app — API routes, preprocessing, model serving
│   ├── digit_recognition_model.h5     # Pre-trained CNN model (Keras HDF5)
│   └── requirements.txt               # Python dependencies
├── frontend/
│   └── index.html                     # Single-page app — canvas, UI, JS logic (44 KB)
├── tests/
│   └── test_app.py                    # End-to-end test suite (Playwright + API)
├── cnn_training.py                        # CNN model training script (MNIST, 99%+ accuracy)
├── hand-written-predictio-website.ipynb   # Jupyter notebook — experimentation
├── Dockerfile                         # Multi-stage Docker build
├── render.yaml                        # Render deployment config
└── .gitignore
```

### Data Flow

```
Canvas Drawing  →  base64 PNG  →  POST /predict
                                      │
                              ┌───────▼────────┐
                              │  Preprocessing  │
                              │  (10-step pipe) │
                              └───────┬────────┘
                                      │
                              ┌───────▼────────┐
                              │   CNN Model     │
                              │   (28×28×1)     │
                              └───────┬────────┘
                                      │
                              { digit, confidence,
                                all_probabilities,
                                inference_time_ms }
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+**
- **pip** (or any Python package manager)

### Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/Kathir-Naveen/Web-app-Handwritten-Digit-Recognition.git
cd Web-app-Handwritten-Digit-Recognition

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Start the server
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** — the frontend is served automatically.

### Docker

```bash
# Build and run
docker build -t neural-digit .
docker run -p 8000:10000 neural-digit

# Or with a custom port
docker run -e PORT=8080 -p 8080:8080 neural-digit
```

---

## 📡 API Reference

### `GET /health`

Returns server health and model status.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "predictions": 42,
  "uptime_since": "2025-02-14T12:30:00"
}
```

### `POST /predict`

Send a base64-encoded canvas image for digit recognition.

**Request:**

```json
{
  "image": "data:image/png;base64,iVBORw0KGgo..."
}
```

**Response:**

```json
{
  "digit": 7,
  "confidence": 0.9982,
  "all_probabilities": {
    "0": 0.0001, "1": 0.0003, "2": 0.0002,
    "3": 0.0001, "4": 0.0001, "5": 0.0002,
    "6": 0.0000, "7": 0.9982, "8": 0.0005, "9": 0.0003
  },
  "inference_time_ms": 12.34
}
```

### `GET /`

Serves the frontend single-page application.

### `GET /docs`

Interactive Swagger UI for the API (auto-generated by FastAPI).

---

## 🧪 Testing

The test suite covers **API endpoints**, **browser UI**, **drawing/prediction flow**, **mobile responsiveness**, and **keyboard shortcuts** using **Playwright**.

```bash
# Install Playwright (one-time)
pip install playwright
playwright install chromium

# Start the server (if not running)
uvicorn backend.main:app --host 0.0.0.0 --port 8000 &

# Run the test suite
python tests/test_app.py
```

### Test Coverage

| Category | Tests |
|----------|-------|
| **API Endpoints** | Health check, prediction with valid/invalid/empty data |
| **Browser UI** | Canvas, buttons, status pill, result card rendering |
| **Drawing & Prediction** | Draw → Predict → Verify digit/confidence/probabilities |
| **Multi-Digit** | 3 different digit shapes with sequential predict/clear |
| **Mobile** | Viewport fit, touch drawing, no horizontal overflow |
| **Keyboard Shortcuts** | `Enter` → predict, `Escape` → clear |

Screenshots are saved to `/tmp/test_screenshots/` for visual inspection.

---

## 🧠 Model Details

| Property | Value |
|----------|-------|
| **Architecture** | Convolutional Neural Network (CNN) |
| **Framework** | TensorFlow / Keras |
| **Dataset** | MNIST (60K train / 10K test) |
| **Test Accuracy** | **99.56%** |
| **Input Shape** | `(28, 28, 1)` — grayscale |
| **Output** | Softmax over 10 classes (digits 0–9) |
| **File** | `backend/digit_recognition_model.h5` |

The full training pipeline is in [`cnn_training.py`](cnn_training.py).

---

## 🌐 Deployment

### Render (Recommended)

This repo includes a [`render.yaml`](render.yaml) for one-click deployment:

1. Push to GitHub
2. Connect the repo on [Render](https://render.com)
3. Render auto-detects the Docker config and deploys

The health check endpoint `/health` is configured for automatic monitoring.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `10000` | Server port (auto-set by Render) |
| `MODEL_PATH` | `backend/digit_recognition_model.h5` | Path to the Keras model file |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | Vanilla HTML/CSS/JS, Canvas API, Orbitron + Rajdhani fonts |
| **Backend** | FastAPI, Uvicorn, Pydantic v2 |
| **ML/AI** | TensorFlow 2.16+, Keras, OpenCV, NumPy |
| **Testing** | Playwright (E2E), urllib (API) |
| **Deployment** | Docker, Render |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Built with 🧠 TensorFlow & ⚡ FastAPI
</p>
