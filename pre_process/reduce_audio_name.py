import os
import re

# === CONFIGURATION ===
input_folder = "./cv-corpus-22.0-delta-2025-06-20/hi/data"       # folder containing your prefixed mp3 files
output_folder = "./cv-corpus-22.0-delta-2025-06-20/hi/data1"   # new folder for renamed clips

os.makedirs(output_folder, exist_ok=True)

# === STEP 1: Get all files ===
files = [f for f in os.listdir(input_folder) if f.endswith('.mp3')]

# === STEP 2: Extract unique client_ids ===
client_ids = []
for f in files:
    match = re.match(r"([0-9a-f]+)_(common_voice_hi_\d+\.mp3)", f)
    if match:
        cid = match.group(1)
        if cid not in client_ids:
            client_ids.append(cid)

# === STEP 3: Create mapping to sequential numbers ===
client_id_map = {cid: f"client_{i+1:03d}" for i, cid in enumerate(client_ids)}

# === STEP 4: Rename and copy ===
for f in files:
    match = re.match(r"([0-9a-f]+)_(common_voice_hi_\d+\.mp3)", f)
    if match:
        cid = match.group(1)
        rest = match.group(2)
        new_cid = client_id_map[cid]
        new_name = f"{new_cid}_{rest}"

        src = os.path.join(input_folder, f)
        dest = os.path.join(output_folder, new_name)
        os.rename(src, dest)  # move + rename
    else:
        print(f"⚠️ Skipping unmatched filename: {f}")

print("✅ Renaming complete!")
print(f"Total unique clients: {len(client_id_map)}")
