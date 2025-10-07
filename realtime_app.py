#Final stablr
# realtime_app.py

import os
import time
import glob
import json
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import joblib

edit_start_time = time.time()


# ── Configuration ─────────────────────────────────────────
IMG_FOLDER   = "images"
MODEL_PATH   = "models/gesture_rf.pkl"
LABELS_PATH  = "models/labels_inv.json"
SAVE_DIR     = "saved"
CAM_W, CAM_H = 640, 480
DEBOUNCE     = 5   # frames to smooth each gesture

os.makedirs(SAVE_DIR, exist_ok=True)

# ── Load images ───────────────────────────────────────────
image_paths = sorted(glob.glob(os.path.join(IMG_FOLDER, "*.*")))
if not image_paths:
    raise SystemExit("No images found in 'images/'")

orig_images = [cv2.imread(p) for p in image_paths]



# ── Load model if available ──────────────────────────────
model, labels_inv = None, {}
if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
    model      = joblib.load(MODEL_PATH)
    labels_inv = json.load(open(LABELS_PATH))
    print(f"[DEBUG] Loaded model + labels: {labels_inv}")
else:
    print("[DEBUG] No ML model found; using geometry fallback")



# ── Helpers ───────────────────────────────────────────────
def normalize_lm(lm):
    pts = np.array([[p.x, p.y, p.z] for p in lm], np.float32)
    pts -= pts[0]  # translate so wrist is origin
    ref = np.linalg.norm(pts[9] - pts[0]) or 1.0
    pts /= ref
    return pts.flatten().reshape(1, -1)

def fallback(lm):
    # simple pinch vs open‐palm
    d = np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
    return "PINCH" if d < 0.05 else "OPEN_PALM"

def save_image(img, idx):
    ts = int(time.time() * 1000)
    fn = f"{idx:02d}_saved_{ts}.png"
    path = os.path.join(SAVE_DIR, fn)
    cv2.imwrite(path, img)
    print(f"[SAVED] {path}")

def draw_brightness_bar(img, brightness, x_offset, y_offset):
    """
    Draws a horizontal 5-block indicator from left to right.
    brightness: -100 … +100
    x_offset, y_offset: coordinates of the LEFT-MOST block
    """
    levels   = 5
    block_w  = 20
    block_h  = 12
    spacing  = 5
    on_blocks = int(round(np.interp(brightness, [-100, 100], [0, levels])))
    on_blocks = np.clip(on_blocks, 0, levels)

    for i in range(levels):
        x1 = x_offset + i * (block_w + spacing)
        y1 = y_offset
        x2 = x1 + block_w
        y2 = y1 + block_h

        color = (0, 255, 0) if i < on_blocks else (50, 50, 50)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (80, 80, 80), 1)




# ── ImageState: transforms + letterbox ────────────────────
class ImageState:
    def __init__(self, img):
        self.orig   = img.copy()
        self.angle  = 0.0
        self.scale  = 1.0
        self.bright = 0.0

    def render(self, tw, th):
        # brightness
        im = np.clip(self.orig.astype(np.int16) + self.bright, 0, 255).astype(np.uint8)
        h, w = im.shape[:2]
        # rotate around center
        M = cv2.getRotationMatrix2D((w/2, h/2), self.angle, 1.0)
        im = cv2.warpAffine(im, M, (w, h))
        # scale
        sf = min(tw / w, th / h) * self.scale
        nw, nh = int(w * sf), int(h * sf)
        im = cv2.resize(im, (nw, nh))
        # crop if zoomed
        if nw > tw or nh > th:
            x0 = (nw - tw) // 2
            y0 = (nh - th) // 2
            return im[y0:y0+th, x0:x0+tw]
        # letterbox pad
        top    = (th - nh) // 2
        bottom = th - nh - top
        left   = (tw - nw) // 2
        right  = tw - nw - left
        return cv2.copyMakeBorder(
            im, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(0,0,0)
        )

# ── Guide Panel ───────────────────────────────────────────
def make_guide(mode):
    lines = {
        "select": [
            "MODE: SELECT",
            "FIST    → Next image",
            "V_SIGN  → Previous",
            "PINCH   → Enter EDIT",
            "",
            "[ESC] Quit"
        ],
        "edit": [
            "MODE: EDIT",
            "V_SIGN    → Rotate",
            "PINCH     → Zoom",
            "OPEN_PALM → Brightness",
            "",
            "[s] Save  [d] Discard" ,
            "[b] Back to SELECT",
            "[ESC] Quit"
        ]
    }[mode]
    panel = np.full((200, 300, 3), 30, np.uint8)
    y = 25
    for L in lines:
        cv2.putText(panel, L, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        y += 25
    return panel

# ── Main Loop ─────────────────────────────────────────────
def main():
    states     = [ImageState(im) for im in orig_images]
    idx, mode  = 0, "select"
    last_g     = None
    last_pin   = None
    hist       = deque(maxlen=DEBOUNCE)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    print("[DEBUG] Running. Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)

        gesture = None
        lm = None

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            mp_draw.draw_landmarks(frame,
                                   res.multi_hand_landmarks[0],
                                   mp.solutions.hands.HAND_CONNECTIONS)

            # classify
            if model:
                feat = normalize_lm(lm)
                pred = model.predict(feat)[0]
                g = labels_inv.get(str(pred), "UNKNOWN")
            else:
                g = fallback(lm)

            hist.append(g)
            gesture = max(set(hist), key=hist.count)

            # SELECT mode
            if mode == "select" and gesture != last_g:
                if gesture == "FIST":
                    idx = (idx + 1) % len(states)
                elif gesture == "V_SIGN":
                    idx = (idx - 1) % len(states)
                elif gesture == "PINCH":
                    mode = "edit"
                last_pin = None

            # EDIT mode
            elif mode == "edit":

                # rotate
                if gesture == "V_SIGN":
                    dx  = lm[9].x - lm[0].x
                    dy  = lm[9].y - lm[0].y
                    states[idx].angle = np.degrees(np.arctan2(dy, dx))

                # pinch zoom
                if gesture == "PINCH":
                    if time.time() - edit_start_time > 5:
                        d = np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
                    if last_pin is not None:
                        ds = (d - last_pin) * 5.0
                        states[idx].scale = np.clip(states[idx].scale + ds, 0.3, 3.0)
                    last_pin = d
                else:
                    last_pin = None

                # brightness
                if gesture == "OPEN_PALM":
                    wy = lm[0].y
                    states[idx].bright = np.clip((0.5 - wy) * 300, -100, 100)

            last_g = gesture

        # render panels
        img_panel = states[idx].render(CAM_W, CAM_H)
        canvas    = np.hstack((frame, img_panel))

        # gesture label
        if gesture:
            cv2.putText(canvas, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if mode == "edit" and gesture == "OPEN_PALM":
            x_off = frame.shape[1] + 10
            y_off = 50
            draw_brightness_bar(
                canvas,
                states[idx].bright,
                x_offset=x_off,
                y_offset=y_off
            )

        # show windows
        cv2.imshow("APP",   canvas)
        cv2.imshow("GUIDE", make_guide(mode))

        # keyboard actions
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):        # save current edited panel
            save_image(states[idx].render(CAM_W, CAM_H), idx)
        elif key == ord('d'):      # discard edits
            states[idx] = ImageState(orig_images[idx])
            print("[DISCARD] Reverted edits.")
        elif mode == "edit" and key == ord('b'):   # BACK to SELECT
            mode = "select"
            print("[MODE] → SELECT")
        elif key == 27:            # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
