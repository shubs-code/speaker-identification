import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import datetime

# -------------------------------
# Configuration
# -------------------------------
SAMPLE_RATE = 16000   # 16 kHz
CHANNELS = 1          # mono
DURATION = 5         # seconds to record
OUTPUT_FILENAME = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

# -------------------------------
# Record audio
# -------------------------------
print(f"🎙️ Recording for {DURATION} seconds at {SAMPLE_RATE} Hz (mono)...")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
sd.wait()  # Wait until recording is finished
print("✅ Recording complete!")

# -------------------------------
# Normalize and save as WAV
# -------------------------------
# Convert float (-1 to 1) → int16 range
audio_int16 = np.int16(audio / np.max(np.abs(audio)) * 32767)
write(OUTPUT_FILENAME, SAMPLE_RATE, audio_int16)

print(f"💾 Saved as: {OUTPUT_FILENAME}")
