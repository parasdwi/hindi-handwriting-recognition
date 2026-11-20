"""Loads the trained model and turns a photo of one handwritten Hindi word into text."""
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageOps

from model_def import IMG_H, IMG_W, build_scrat_net

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "v4_best.weights.h5")
CHARSET_PATH = os.path.join(HERE, "charset.json")
MAX_SIDE = 1600 

def pil_to_gray(img: Image.Image) -> np.ndarray:
    """Fix phone rotation (EXIF), remove transparency, convert to grayscale, shrink."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        white = Image.new("RGB", img.size, "white")
        white.paste(img, mask=img.split()[-1])
        img = white
    gray = np.array(img.convert("L"))
    h, w = gray.shape
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_AREA)
    return gray


def normalize_lighting(gray: np.ndarray) -> np.ndarray:
    """Estimate the paper/shadow background and remove it (white paper, dark ink)."""
    k = max(3, (min(gray.shape) // 30) | 1)  
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((k, k), np.uint8)), k)
    diff = cv2.absdiff(gray, bg)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    return (255 - diff).astype(np.uint8)


def crop_to_ink(gray: np.ndarray) -> np.ndarray:
    """Crop tightly around the dark ink, with a small margin."""
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if mask.mean() > 127:
        mask = 255 - mask
    ys, xs = np.where(mask > 0)
    if len(xs) < 20:
        return gray

    x_lo, x_hi = np.percentile(xs, [0.5, 99.5]).astype(int)
    y_lo, y_hi = np.percentile(ys, [0.5, 99.5]).astype(int)
    h, w = gray.shape
    px = int(0.08 * (x_hi - x_lo + 1))
    py = int(0.25 * (y_hi - y_lo + 1))
    x0, x1 = max(0, x_lo - px), min(w, x_hi + px + 1)
    y0, y1 = max(0, y_lo - py), min(h, y_hi + py + 1)
    return gray[y0:y1, x0:x1]


def autocontrast(gray: np.ndarray) -> np.ndarray:
    """Stretch brightness so the 1st..99th percentiles span 0..255.
    The v3/v4 models were trained on images prepared exactly this way, so it must be applied here too."""
    lo, hi = np.percentile(gray, [1, 99])
    if hi - lo < 10:
        return gray
    out = (gray.astype(np.float32) - lo) * 255.0 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def to_model_input(gray: np.ndarray) -> np.ndarray:
    """Same steps as training: height 32, keep proportions, invert, pad with black to 128."""
    h, w = gray.shape
    new_w = min(IMG_W, max(1, round(w * IMG_H / h)))
    small = cv2.resize(gray, (new_w, IMG_H), interpolation=cv2.INTER_AREA)
    small = 255 - small
    canvas = np.zeros((IMG_H, IMG_W), dtype=np.uint8)
    canvas[:, :new_w] = small
    return canvas

class Recognizer:
    def __init__(self, weights_path=WEIGHTS_PATH, charset_path=CHARSET_PATH):
        with open(charset_path, encoding="utf-8") as f:
            self.chars = json.load(f)["chars"]
        self.blank = len(self.chars)  
        self.model = build_scrat_net(len(self.chars) + 1)
        self.model.load_weights(weights_path)

    def decode(self, probs: np.ndarray) -> str:
        """Greedy CTC decoding: best class per step, merge repeats, drop blanks."""
        best = np.argmax(probs, axis=-1)
        out, prev = [], -1
        for k in best:
            if k != prev and k != self.blank:
                out.append(self.chars[int(k)])
            prev = k
        return "".join(out)

    def predict(self, img: Image.Image, crop: bool = True, fix_light: bool = False):
        gray = pil_to_gray(img)
        if fix_light:
            gray = normalize_lighting(gray)
        if crop:
            gray = crop_to_ink(gray)
        gray = autocontrast(gray)
        x = to_model_input(gray)
        batch = x[None, :, :, None].astype("float32") / 255.0
        probs = self.model(batch, training=False).numpy()[0]
        return self.decode(probs), gray, x
