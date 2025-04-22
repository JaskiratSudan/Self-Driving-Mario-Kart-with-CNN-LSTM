#!/usr/bin/env python3
import time
import logging
from collections import deque

import numpy as np
from mss import mss
from tensorflow.keras.models import load_model
from train import customized_loss
from skimage.transform import resize
from pynput.keyboard import Controller, Key

# ---- CONFIG ----
MODEL_FILE = "model_weights.keras"
FPS = 20
STEER_DEADZONE   = 0.01    # ignore low‑confidence steering
THROTTLE_DEADZONE= 0.1     # require >0.1 to throttle
BRAKE_DEADZONE   = 0.1     # require >0.1 to brake
SEQ_LEN          = 3      # must match your training script

# Key mappings (must match your training script)
THROTTLE_KEY    = Key.shift   # accelerate
BRAKE_KEY       = Key.ctrl    # brake/reverse
STEER_LEFT_KEY  = Key.left    # steer left
STEER_RIGHT_KEY = Key.right   # steer right

# Screen capture region (adjust if window not at origin)
REGION = {
    "top":    0,
    "left":   0,
    "width":  640,
    "height": 480
}

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Load model with custom loss (though loss isn't used at inference)
model = load_model(MODEL_FILE, custom_objects={"customized_loss": customized_loss})
kbd   = Controller()

# Input dimensions
IMG_H, IMG_W, _ = 64, 64, 3

def preprocess(img_bgra):
    """
    Convert BGRA to RGB, resize to model input, normalize to [0,1].
    Returns a single frame of shape (H, W, 3).
    """
    # drop alpha, convert BGR->RGB
    img_rgb = img_bgra[..., :3][..., ::-1]
    img_rs  = resize(img_rgb, (IMG_H, IMG_W, 3), preserve_range=True)
    return img_rs.astype("float32") / 255.0

def apply_action(pred):
    """
    Apply throttle, brake, and steering based on prediction.
    `pred` is [f, l, r, b].
    """
    f_pred, l_pred, r_pred, b_pred = pred

    # release all keys
    for key in (THROTTLE_KEY, BRAKE_KEY, STEER_LEFT_KEY, STEER_RIGHT_KEY):
        kbd.release(key)

    pressed = []
    # Throttle
    if f_pred > THROTTLE_DEADZONE:
        kbd.press(THROTTLE_KEY)
        pressed.append('Throttle')
    # Brake
    if b_pred > BRAKE_DEADZONE:
        kbd.press(BRAKE_KEY)
        pressed.append('Brake')
    # Steering: choose stronger side
    if max(l_pred, r_pred) > STEER_DEADZONE:
        if l_pred > r_pred:
            kbd.press(STEER_LEFT_KEY)
            pressed.append('Left')
        else:
            kbd.press(STEER_RIGHT_KEY)
            pressed.append('Right')

    # log
    print(f"Preds ➜ f={f_pred:.3f}, l={l_pred:.3f}, r={r_pred:.3f}, b={b_pred:.3f}")
    print("Keys pressed:", ", ".join(pressed) if pressed else "None")

def main():
    print("⚠️  Focus the Mupen64Plus window now (Shift/Ctrl/Arrows).")
    time.sleep(2)

    sct      = mss()
    interval = 1.0 / FPS

    # rolling buffer of last SEQ_LEN frames
    frame_buffer = deque(maxlen=SEQ_LEN)

    # seed buffer by capturing one frame and repeating
    first = np.array(sct.grab(REGION))
    f_proc = preprocess(first)
    for _ in range(SEQ_LEN):
        frame_buffer.append(f_proc)

    try:
        while True:
            start = time.time()

            img = np.array(sct.grab(REGION))
            f_proc = preprocess(img)
            frame_buffer.append(f_proc)

            # stack into shape (1, SEQ_LEN, H, W, D)
            seq = np.stack(frame_buffer, axis=0)[None, ...]
            pred = model.predict(seq, verbose=0)[0]
            apply_action(pred)

            # maintain target FPS
            elapsed = time.time() - start
            time.sleep(max(0, interval - elapsed))

    except KeyboardInterrupt:
        pass

    finally:
        # make sure all keys are released
        for key in (THROTTLE_KEY, BRAKE_KEY, STEER_LEFT_KEY, STEER_RIGHT_KEY):
            kbd.release(key)
        logging.info("Exiting: released all keys.")

if __name__ == "__main__":
    main()