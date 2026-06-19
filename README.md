# Recycling Lab Tycoon — AI Backend & Model Training

An AI-powered simulation management tycoon game. Players interact with the game by holding up real-world garbage items in front of their webcam. A Python WebSocket Server handles the real-time camera feed, performs inference across 5 deep learning models in an ensemble, and sends the consensus classification back to the Unity game client.

---

## 📁 Repository Structure

```
D:\Garbage-Classification\
├── models/
│   ├── models-weight/
│   │   ├── resnet50_best.pth            # Trained ResNet-50 weights
│   │   ├── efficientnet_b0_best.pth     # Trained EfficientNet-B0 weights
│   │   ├── efficientnet_v2_s_best.pth   # Trained EfficientNet-V2-S weights
│   │   ├── mobilenet_v3_large_best.pth  # Trained MobileNet-V3-Large weights
│   │   └── convnext_tiny_best.pth       # Trained ConvNeXt-Tiny weights
│   └── garbage-classification.ipynb     # Jupyter Notebook for model training
├── server/
│   ├── camera.py             # Thread-safe OpenCV camera capture (10 FPS)
│   ├── model_loader.py       # Model initialization and preprocessing pipeline
│   ├── ensemble.py           # Majority voting ensemble decision logic
│   ├── websocket_server.py   # Asyncio WebSocket server for broadcasting
│   └── run_server.py         # Application entry point with CLI options
├── .gitignore                # Git ignore configuration
└── README.md                 # Project guide (this file)
```

---

## ⚡ Quick Start: AI Backend Server Setup

Follow these steps to set up and run the real-time AI inference backend.

### 1. Requirements & Prerequisites
Ensure you have Python 3.10+ installed. Install the necessary dependencies:

```bash
pip install torch torchvision opencv-python websockets pillow numpy
```

### 2. Run the Server
The entry-point is `server/run_server.py`. By default, it automatically scans for weights in `models/models-weight`, opens camera index `0`, and listens on `ws://localhost:8765` at `10 FPS`.

#### Default Launch:
```bash
python server/run_server.py
```

#### Custom Launch Options:
You can configure the server using Command Line Arguments:
```bash
# Run on GPU (cuda), set custom port, camera index, and slower frame rate
python server/run_server.py --device cuda --port 9000 --camera 1 --fps 5 --log-level DEBUG
```

*Arguments:*
- `--weights-dir`: Path to the directory containing the five `.pth` files.
- `--host`: Server host address (default: `localhost`).
- `--port`: WebSocket port (default: `8765`).
- `--camera`: OpenCV webcam index (default: `0`).
- `--fps`: Target loop speed / frames per second (default: `10`).
- `--device`: Target device `cpu` or `cuda`. If omitted, it auto-detects GPU availability.
- `--log-level`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`).

### 3. API Output Format
The WebSocket server broadcasts a JSON string containing the ensemble consensus, individual votes, and latency metrics to all connected clients every cycle:

```json
{
  "detected_item": "plastic",
  "confidence": 0.8950,
  "all_votes": ["plastic", "plastic", "cardboard", "plastic", "white-glass"],
  "inference_ms": 124.5
}
```

---

## 🏋️ Model Training Setup

The classification system is trained on 12 garbage categories using a Jupyter Notebook.

### 1. The Dataset
The models are trained using the [Garbage Classification Dataset](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) by Mostafa Abla on Kaggle.

#### Target Classes (12 categories, alphabetized):
`battery`, `biological`, `brown-glass`, `cardboard`, `clothes`, `green-glass`, `metal`, `paper`, `plastic`, `shoes`, `trash`, `white-glass`.

### 2. Training via Jupyter Notebook
You can run the training locally or directly on Kaggle.

#### Running on Kaggle (Recommended for Free GPUs):
1. Create a new notebook on Kaggle.
2. Add the dataset: `mostafaabla/garbage-classification`.
3. Import the `models/garbage-classification.ipynb` notebook file.
4. Set the Accelerator to **GPU (T4 x2 or P100)**.
5. Execute the cells sequentially.

#### Running Locally:
1. Download the dataset from Kaggle and extract it.
2. Start your Jupyter server:
   ```bash
   jupyter notebook
   ```
3. Open `models/garbage-classification.ipynb`.
4. Update the `dataset_folder` path in **Cell 3** to point to your local dataset directory:
   ```python
   # Example local path
   dataset_folder = 'path/to/extracted/garbage_classification'
   ```
5. Run all cells to begin model training. The script will train all 5 models sequentially and output their best weight files (e.g., `resnet50_best.pth`).
6. Copy the generated weight files (`*_best.pth`) into your `models/models-weight/` directory for the server to use.

---

## 🤝 Unity Game Integration Guide

1. **WebSocket Connection**: Connect to the server from Unity using a WebSocket client (such as `NativeWebSocket` package) pointing to `ws://localhost:8765`.
2. **Match Logic**: 
   - Parse the incoming JSON message.
   - Match `detected_item` with your `currentRequiredTrash`.
   - Ensure `confidence` is above a safety threshold (e.g., `> 0.65`) to avoid noise or cheating.
3. **Anti-Spam / Debounce**: Lock the processing of new frames for `1.0` to `2.0` seconds immediately after a successful match to give the player time to clear the camera view.