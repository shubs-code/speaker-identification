import librosa
import numpy as np
import tensorflow as tf
import pickle

# Load model and encoder
model = tf.keras.models.load_model("speaker_identification_model.h5")
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

SAMPLE_RATE = 16000
DURATION = 3  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
N_MFCC = 40

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

def predict_speaker(file_path):
    mfcc = extract_mfcc(file_path)
    if mfcc is None:
        return "Error processing file"
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    pred = model.predict(mfcc)
    label = encoder.inverse_transform([np.argmax(pred)])[0]
    return label

print(predict_speaker("./cv-corpus-22.0-delta-2025-06-20/hi/data1/client_006_common_voice_hi_43388577.mp3"))
