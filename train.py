#!/usr/bin/env python3
import sys
import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, TimeDistributed, LSTM
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    TimeDistributed, LSTM, Input
)
from tensorflow.keras import optimizers, backend as K
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# --- CONFIGURATION ---
IMG_H, IMG_W, IMG_D     = 64, 64, 3
SEQ_LEN                 = 3
OUT_SHAPE               = 4           # forward, left, right, brake
EPOCHS                  = 100
BATCH_SIZE              = 64
VALIDATION_SPLIT        = 0.1

# Balancing hyperparameters
LEFT_SAMPLE_WEIGHT      = 3.0
RIGHT_SAMPLE_WEIGHT     = 3.0
LEFT_LOSS_WEIGHT        = 3.0
RIGHT_LOSS_WEIGHT       = 3.0

CSV_FILENAME            = 'data.csv'
CSV_MIN_FIELDS          = OUT_SHAPE + 1  # image_path + 4 values

def customized_loss(y_true, y_pred):
    """Weighted Euclidean‐style loss."""
    w = K.constant([1.0, LEFT_LOSS_WEIGHT, RIGHT_LOSS_WEIGHT, 1.0],
                   dtype=tf.float32)
    diff_sq  = K.square(y_pred - y_true)
    weighted = w * diff_sq
    return K.sqrt(K.sum(weighted, axis=-1))

def create_cnn_lstm_model(seq_len, img_h, img_w, img_d, out_shape):
    """Builds TimeDistributed CNN → LSTM model with stable sigmoid outputs."""
    cnn = Sequential([
        Input(shape=(img_h, img_w, img_d)),
        Conv2D(24, (5,5), strides=2, padding='same', activation='relu'),
        MaxPooling2D((2,2)),

        Conv2D(36, (5,5), strides=2, padding='same', activation='relu'),
        MaxPooling2D((2,2)),

        Conv2D(48, (5,5), strides=2, padding='same', activation='relu'),
        Conv2D(64, (3,3), padding='same', activation='relu'),
        Conv2D(64, (3,3), padding='same', activation='relu'),

        Flatten(),
        Dense(128, activation='relu'),
    ])

    seq_in = Input(shape=(seq_len, img_h, img_w, img_d))
    x = TimeDistributed(cnn)(seq_in)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)

    # ← use sigmoid so outputs ∈ [0,1]
    out = Dense(out_shape, activation='sigmoid')(x)

    model = Model(seq_in, out)
    # ← clip gradients and/or lower LR to avoid blowups
    opt = optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)
    model.compile(optimizer=opt, loss=customized_loss)
    return model
    
    # Sequence input
    seq_in = Input(shape=(seq_len, img_h, img_w, img_d))
    x = TimeDistributed(cnn)(seq_in)
    x = LSTM(64)(x)
    x = Dropout(0.2)(x)
    out = Dense(out_shape, activation='relu')(x)

    model = Model(inputs=seq_in, outputs=out)
    model.compile(optimizer=optimizers.Adam(), loss=customized_loss)
    return model

def load_data_from_csvs(csv_paths):
    """Loads every frame listed in csv_paths, returns sliding windows + labels."""
    image_files, labels = [], []
    for csv_path in csv_paths:
        if not os.path.isfile(csv_path):
            print(f"Warning: CSV not found, skipping: {csv_path}")
            continue
        with open(csv_path, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < CSV_MIN_FIELDS:
                    continue
                image_files.append(row[0])
                labels.append([float(v) for v in row[1:CSV_MIN_FIELDS]])

    if not image_files:
        print("Error: no images found in any CSV.")
        sys.exit(1)

    N = len(image_files)
    X = np.zeros((N, IMG_H, IMG_W, IMG_D), dtype=np.float32)
    for i, fp in enumerate(image_files):
        try:
            img = load_img(fp, target_size=(IMG_H, IMG_W))
            X[i] = img_to_array(img) / 255.0
        except Exception as e:
            print(f"Error loading {fp}: {e}")
            sys.exit(1)

    y = np.array(labels, dtype=np.float32)

    # slide window of length SEQ_LEN
    seq_count = N - SEQ_LEN + 1
    X_seq = np.zeros((seq_count, SEQ_LEN, IMG_H, IMG_W, IMG_D),
                     dtype=np.float32)
    y_seq = np.zeros((seq_count, OUT_SHAPE), dtype=np.float32)

    for i in range(seq_count):
        X_seq[i] = X[i:i+SEQ_LEN]
        y_seq[i] = y[i+SEQ_LEN-1]

    return X_seq, y_seq

if __name__ == "__main__":
    usage = (
        "Usage:\n"
        "  # from scratch:\n"
        "    python3 train.py <samples_folder>\n"
        "  # load & retrain:\n"
        "    python3 train.py <samples_folder> <existing_model.keras>\n"
    )
    if len(sys.argv) not in (2, 3):
        print(usage)
        sys.exit(1)

    samples_root = sys.argv[1]
    if not os.path.isdir(samples_root):
        print(f"Error: not a folder: {samples_root}")
        sys.exit(1)

    existing_model_path = None
    if len(sys.argv) == 3:
        existing_model_path = sys.argv[2]
        if not os.path.isfile(existing_model_path):
            print(f"Error: model file not found: {existing_model_path}")
            sys.exit(1)

    csv_paths = []
    for sub in os.listdir(samples_root):
        p = os.path.join(samples_root, sub, CSV_FILENAME)
        if os.path.isfile(p):
            csv_paths.append(p)
    if not csv_paths:
        print("Error: no subfolders with", CSV_FILENAME)
        sys.exit(1)

    x_train, y_train = load_data_from_csvs(csv_paths)
    print(f"→ {x_train.shape[0]} sequences of {SEQ_LEN} frames loaded")
    print("  input shape:", x_train.shape[1:], " output shape:", y_train.shape[1:])

    sample_weights = np.ones(x_train.shape[0], dtype=np.float32)
    left_idx  = np.where(y_train[:,1] > 0)[0]
    right_idx = np.where(y_train[:,2] > 0)[0]
    if left_idx.size:
        sample_weights[left_idx]  *= LEFT_SAMPLE_WEIGHT
        print(f"→ Left turns ×{LEFT_SAMPLE_WEIGHT}")
    if right_idx.size:
        sample_weights[right_idx] *= RIGHT_SAMPLE_WEIGHT
        print(f"→ Right turns ×{RIGHT_SAMPLE_WEIGHT}")

    if existing_model_path:
        print(f"Loading existing model from {existing_model_path} …")
        model = load_model(existing_model_path,
                           custom_objects={"customized_loss": customized_loss})
        save_path = 'models/retrained_model_best.keras'
    else:
        model = create_cnn_lstm_model(SEQ_LEN, IMG_H, IMG_W, IMG_D, OUT_SHAPE)
        save_path = 'model_weights.keras'

    checkpoint = ModelCheckpoint(
        save_path, monitor='val_loss', save_best_only=True,
        mode='min', verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss', patience=10,
        restore_best_weights=True, verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2,
        patience=5, min_lr=1e-3, verbose=1
    )

    history = model.fit(
        x_train, y_train,
        sample_weight=sample_weights,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        shuffle=True,
        validation_split=VALIDATION_SPLIT,
        callbacks=[checkpoint, reduce_lr, early_stop]
    )

    print(f"✅ Training complete!  Best model saved to: {save_path}")