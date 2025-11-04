import os
from pydub import AudioSegment

# === CONFIGURATION ===
input_folder = "./cv-corpus-22.0-delta-2025-06-20/hi/data1"     # folder with your .mp3 files
output_folder = "./cv-corpus-22.0-delta-2025-06-20/hi/data2"      # folder for 3-sec .wav chunks
segment_length = 3 * 1000             # 3 seconds in milliseconds

os.makedirs(output_folder, exist_ok=True)

# === PROCESS EACH MP3 ===
for filename in os.listdir(input_folder):
    if not filename.endswith(".mp3"):
        continue

    input_path = os.path.join(input_folder, filename)

    try:
        # Load audio
        audio = AudioSegment.from_file(input_path, format="mp3")
        duration = len(audio)

        # Split into 3-second chunks
        num_segments = duration // segment_length + (1 if duration % segment_length != 0 else 0)
        base_name = os.path.splitext(filename)[0]

        for i in range(num_segments):
            start_time = i * segment_length
            end_time = min((i + 1) * segment_length, duration)
            segment = audio[start_time:end_time]

            # Convert to WAV (16 kHz mono recommended for speech)
            segment = segment.set_frame_rate(16000).set_channels(1)

            output_filename = f"{base_name}_part{i}.wav"
            output_path = os.path.join(output_folder, output_filename)
            segment.export(output_path, format="wav")

        print(f"✅ Processed {filename} → {num_segments} parts")

    except Exception as e:
        print(f"⚠️ Error processing {filename}: {e}")

print("🎧 All files segmented successfully.")
