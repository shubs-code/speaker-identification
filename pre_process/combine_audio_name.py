import os
import shutil
import pandas as pd

# === CONFIGURATION ===
tsv_path = "./cv-corpus-22.0-delta-2025-06-20/hi/other.tsv"         # path to your TSV file
clips_folder = "./cv-corpus-22.0-delta-2025-06-20/hi/clips"            # folder with your original mp3s
output_folder = "./cv-corpus-22.0-delta-2025-06-20/hi/data"   # folder where renamed files will be saved

# === STEP 1: Load the TSV ===
df = pd.read_csv(tsv_path, sep='\t')

# === STEP 2: Make sure output folder exists ===
os.makedirs(output_folder, exist_ok=True)

# === STEP 3: Loop through rows and rename ===
for _, row in df.iterrows():
    client_id = str(row['client_id'])
    filename = str(row['path'])

    src_path = os.path.join(clips_folder, filename)
    new_filename = f"{client_id}_{filename}"
    dest_path = os.path.join(output_folder, new_filename)

    # Copy only if the source file exists
    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)
    else:
        print(f"⚠️ File not found: {src_path}")

print("✅ All available clips have been copied and renamed.")
