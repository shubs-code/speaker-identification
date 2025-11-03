import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# === 1. Load 10-second audio sample ===
# Change "audio.wav" to your filename
file_path = "sample.wav"

# sr=16000 ensures 16 kHz mono, duration=10 trims/pads to 10 s
y, sr = librosa.load(file_path, sr=16000)
print(f"Loaded audio: {len(y)/sr:.2f} sec, Sample rate: {sr}")

# === 2. Compute MFCC, Delta, and Delta-Delta ===
n_mfcc = 40
mfcc = librosa.feature.mfcc(
    y=y, sr=sr, n_mfcc=n_mfcc,
    n_fft=400, hop_length=160, win_length=400
)

# First and second derivatives (Δ and ΔΔ)
delta = librosa.feature.delta(mfcc)
delta2 = librosa.feature.delta(mfcc, order=2)

# Combine all (stack along feature axis)
mfcc_features = np.vstack([ mfcc, delta, delta2])
print("MFCC feature shape:", mfcc_features.shape)  # (120, time_frames)

# === 3. Visualize ===
plt.figure(figsize=(10, 5))
librosa.display.specshow(mfcc_features, x_axis='time', sr=sr)
plt.colorbar(label='Feature Value')
plt.title('MFCC + Δ + ΔΔ (120 coefficients)')
plt.tight_layout()
plt.show()

# === 4. Save features ===
np.save("audio_mfcc_features.npy", mfcc_features)
print("Saved features to audio_mfcc_features.npy")
