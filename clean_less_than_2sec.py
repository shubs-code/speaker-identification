import os
from pydub import AudioSegment

# === CONFIGURATION ===
folder = "/media/shrubex/f6b29291-706a-4909-a36a-5a835513826a/research/cv-corpus-22.0-delta-2025-06-20/hi/data2"
min_duration_ms = 2000     # 2 seconds in milliseconds

count_deleted = 0
count_kept = 0

for filename in os.listdir(folder):
    if not filename.endswith(".wav"):
        continue

    path = os.path.join(folder, filename)

    try:
        audio = AudioSegment.from_file(path)
        duration = len(audio)

        if duration < min_duration_ms:
            os.remove(path)
            count_deleted += 1
        else:
            count_kept += 1

    except Exception as e:
        print(f"⚠️ Error reading {filename}: {e}")

print(f"🗑️ Deleted {count_deleted} short files (<2s)")
print(f"✅ Kept {count_kept} valid files (≥2s)")
