import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# ---------------------------
# CONFIGURATION
# ---------------------------
DATASET_PATH = "./cv-corpus-22.0-delta-2025-06-20/hi/data2"  # <-- change this
SAMPLE_RATE = 16000
DURATION = 3  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MFCC = 40

# ---------------------------
# FEATURE EXTRACTION
# ---------------------------
def extract_mfcc(file_path, sr=SAMPLE_RATE, duration=DURATION, n_mfcc=N_MFCC):
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        # Pad or trim to exactly 3 seconds
        if len(audio) < sr * duration:
            pad_width = sr * duration - len(audio)
            audio = np.pad(audio, (0, int(pad_width)), mode='constant')
        else:
            audio = audio[:sr * duration]
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return mfcc
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# ---------------------------
# LOAD DATA
# ---------------------------
X, y = [], []

for filename in os.listdir(DATASET_PATH):
    if filename.endswith(".wav"):
        file_path = os.path.join(DATASET_PATH, filename)
        mfcc = extract_mfcc(file_path)
        if mfcc is not None:
            X.append(mfcc)
            # Extract label (client name)
            label = filename.split("_")[0] + "_" + filename.split("_")[1]  # e.g., client_001
            y.append(label)

X = np.array(X)
y = np.array(y)

# Add a channel dimension for CNN
X = X[..., np.newaxis]

print("Data loaded:", X.shape, "Labels:", len(y))

# ---------------------------
# ENCODE LABELS
# ---------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))
y_encoded = tf.keras.utils.to_categorical(y_encoded, num_classes)

# ---------------------------
# TRAIN/TEST SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# ---------------------------
# CNN MODEL FOR SPEAKER ID
# ---------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=X_train.shape[1:]),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(),

    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------------------
# TRAIN MODEL
# ---------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=60,
    batch_size=10
)

# ---------------------------
# SAVE MODEL AND ENCODER
# ---------------------------
model.save("speaker_identification_model.h5")
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)

print("✅ Training complete and model saved.")
