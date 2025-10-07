## Gesture App

Simple real‑time hand gesture image browser and lightweight editor.

Main scripts:
- `capture_data.py` – record hand landmark samples with MediaPipe
- `train_classifier.py` – train a RandomForest model on the captured CSV
- `realtime_app.py` – run the live app: navigate images and edit (rotate / zoom / brightness)

Folders created at runtime (ignored in Git): `dataset/`, `models/`, `images/`, `saved/`.

Basic usage (PowerShell):
```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt

# collect data (press o f p v to label; ESC to quit)
python capture_data.py

# train model
python train_classifier.py

# place some .jpg or .png files in images/ then run
python realtime_app.py
```

Gestures (default):
OPEN_PALM (brightness in edit), FIST (next), V_SIGN (previous / rotate), PINCH (enter edit / zoom).

License: MIT (see `LICENSE`).
