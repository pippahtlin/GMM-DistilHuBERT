# Due to the crazily slow uploading, I have to run this locally
import os
import shutil
import random
from datasets import load_dataset
from pathlib import Path
from tqdm import tqdm

SRC_ROOT = Path("/your/full/train-clean-360")      # Original full dataset
DST_ROOT = Path("/your/export/train-clean-150")     # Output subset path

# Load and shuffle 90,000 examples ≈ 150 hours
dataset = load_dataset("librispeech_asr", "clean", split="train.360", cache_dir="/your/cache") 
subset = dataset.shuffle(seed=42).select(range(90000))

# Track which files are copied
copied_transcripts = set()

for example in tqdm(subset, total=len(subset)):
    file_path = Path(example["file"])  # e.g., /your/full/train-clean-360/84/121123/84-121123-0000.flac
    rel_path = file_path.relative_to(SRC_ROOT)
    dst_path = DST_ROOT / rel_path

    # Copy audio file
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, dst_path)

    # Copy corresponding .trans.txt once per chapter
    chapter_dir = file_path.parent
    trans_file = chapter_dir / f"{chapter_dir.name}.trans.txt"
    rel_trans_path = trans_file.relative_to(SRC_ROOT)
    dst_trans_path = DST_ROOT / rel_trans_path

    if trans_file.exists() and str(rel_trans_path) not in copied_transcripts:
        shutil.copy2(trans_file, dst_trans_path)
        copied_transcripts.add(str(rel_trans_path))

print("✅ Finished exporting train-clean-150 in original LibriSpeech format.")
