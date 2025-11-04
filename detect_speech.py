import os
import torch
import torchaudio
from tqdm import tqdm

# === CONFIG ===
folder = "./cv-corpus-22.0-delta-2025-06-20/hi/data2"   # Folder containing your WAV files
sample_rate = 16000         # Expected sample rate for the model
threshold = 0.3             # Minimum proportion of speech frames to consider as 'speech'

# === Load Silero VAD model ===
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
(get_speech_timestamps, _, read_audio, _, collect_chunks) = utils

nonspeech_files = []

# === Iterate through files ===
for filename in tqdm(os.listdir(folder)):
    if not filename.endswith(".wav"):
        continue

    path = os.path.join(folder, filename)
    try:
        wav, sr = torchaudio.load(path)
        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)

        wav = wav.squeeze()
        speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=sample_rate)

        # Check fraction of speech in total duration
        speech_duration = sum([seg['end'] - seg['start'] for seg in speech_timestamps]) / sample_rate
        total_duration = wav.shape[0] / sample_rate
        speech_ratio = speech_duration / total_duration if total_duration > 0 else 0

        if speech_ratio < threshold:
            nonspeech_files.append(filename)

    except Exception as e:
        print(f"⚠️ Error with {filename}: {e}")

# === Summary ===
print("\n🧩 Possible non-speech / silent clips:")
for f in nonspeech_files:
    print(" -", f)
print(f"\nTotal flagged: {len(nonspeech_files)}")
