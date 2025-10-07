"""Utility script to prepare the local workspace.

Creates the expected directory structure and (optionally) a placeholder image
so the realtime app does not immediately exit when no images are present.

Usage:
  python setup_folders.py            # just create folders
  python setup_folders.py --demo     # also create a demo test image
"""
from __future__ import annotations
import argparse
import os
import cv2
import numpy as np

FOLDERS = [
    "dataset",
    "models",
    "images",
    "saved",
]

def ensure_dirs():
    for d in FOLDERS:
        os.makedirs(d, exist_ok=True)
        keep = os.path.join(d, ".gitkeep")
        if not os.path.exists(keep):
            # Do not overwrite if user created one already
            try:
                with open(keep, "w", encoding="utf-8") as f:
                    f.write("Placeholder to retain folder in Git history.\n")
            except OSError:
                pass

def create_demo_image():
    path = os.path.join("images", "demo_placeholder.png")
    if os.path.exists(path):
        return path
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, "Gesture App Demo", (50,180), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0,255,255), 3, cv2.LINE_AA)
    cv2.putText(img, "Add your own images to /images", (40,300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 1)
    cv2.imwrite(path, img)
    return path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="create a demo image if images folder is empty")
    args = parser.parse_args()

    ensure_dirs()

    created = None
    if args.demo:
        if not any(f.lower().endswith((".png",".jpg",".jpeg")) for f in os.listdir("images")):
            created = create_demo_image()

    print("[OK] Folders verified: " + ", ".join(FOLDERS))
    if created:
        print(f"[OK] Demo image created: {created}")
    elif args.demo:
        print("[SKIP] Images folder already contains user images; no demo created.")

if __name__ == "__main__":
    main()
