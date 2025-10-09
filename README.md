## Gesture App

A small, beginner‑friendly project that uses your webcam and MediaPipe Hands to browse and lightly edit images using hand gestures. You can rotate, zoom, and adjust brightness hands‑free. It also includes simple scripts to collect training data and train a tiny classifier, but the app can still run with a basic fallback if you don’t have a model yet.

### What you can do
- Navigate images (Next/Previous)
- Enter Edit mode with a gesture
- Rotate, Zoom, and change Brightness using hand poses
- Save the edited view as a new image

---

## Requirements
- Windows 10/11 (tested), a webcam
- Python 3.10–3.12 recommended
- The packages listed in `requirements.txt`

Note: The app uses OpenCV and MediaPipe with prebuilt wheels for Windows, so setup should be straightforward.

---

## Quick start (Windows PowerShell)

```powershell
# 1) Create and activate a virtual environment
python -m venv venv
./venv/Scripts/Activate.ps1

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Put a few .jpg/.png files into the images/ folder

# 4) Run the real-time app
python realtime_app.py
```

If you don’t have a trained model yet, the app will print a debug message and use a simple fallback (it still recognizes basic gestures like PINCH/OPEN_PALM). For the full experience, see “Collect data and train the model” below.

---

## Collect data and train the model (optional but recommended)

1) Record hand landmark samples
```powershell
python capture_data.py
```
Keys while recording:
- o → label as OPEN_PALM
- f → label as FIST
- p → label as PINCH
- v → label as V_SIGN
- ESC → finish

This saves feature vectors to `dataset/hand_gesture_landmarks.csv`.

2) Train the classifier
```powershell
python train_classifier.py
```
What it does:
- Splits the dataset, trains a RandomForest
- Saves the model to `models/gesture_rf.pkl`
- Saves label mapping to `models/labels_inv.json`

---

## Gestures and controls

When running `realtime_app.py`:
- SELECT mode
	- FIST → Next image
	- V_SIGN → Previous image
	- PINCH → Enter EDIT mode

- EDIT mode
	- V_SIGN → Rotate image (based on hand direction)
	- PINCH → Zoom in/out (pinch distance)
	- OPEN_PALM → Adjust brightness (hand height)
	- s → Save the edited view into `saved/`
	- d → Discard edits (revert to original)
	- b → Back to SELECT mode
	- ESC → Quit

Tip: A guide panel window shows the active mode and available actions.

---

## Project structure
- `realtime_app.py` — Run the live gesture app
- `capture_data.py` — Record hand landmarks and label them
- `train_classifier.py` — Train a RandomForest model
- `dataset/` — CSV data lives here (ignored by Git, `.gitkeep` inside)
- `models/` — Trained model and labels live here (ignored by Git, `.gitkeep` inside)
- `images/` — Put your input images here (ignored by Git)
- `saved/` — App writes saved edited images here (ignored by Git)

---

## Troubleshooting
• Window shows but camera feed is black
- Another app may be using the webcam. Close it and retry.
- Try camera index 1 or 2 by editing `cv2.VideoCapture(0)` in `realtime_app.py`.

• pip install fails or is slow
- Ensure you’re using a fresh venv and latest pip: `python -m pip install --upgrade pip`.
- On restricted networks, `pip` may need `--trusted-host` or a different index mirror.

• Low FPS or high CPU
- Reduce camera resolution in `realtime_app.py` (CAM_W, CAM_H).
- Ensure only one image window is visible and other heavy apps are closed.

• No model found message
- That’s okay. The app uses a simple fallback for basic gestures. For best results, follow the training steps above.

---

## License
MIT — see `LICENSE`.
