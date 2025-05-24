import os
import soundfile as sf
import pandas as pd
import json
from datasets import load_from_disk, Dataset
from textgrid import TextGrid

REPO_ROOT = "/mnt/scratch/pippalin2/jupyter/GMM-DistilHuBERT"

DATASET_PATH = os.path.join(REPO_ROOT, "data/hf_librispeech_clean100")
TEXTGRID_INPUT_DIR = os.path.join(REPO_ROOT, "mfa/librispeech_textgrids")
OUTPUT_DATASET_DIR = os.path.join(REPO_ROOT, "data/hf_librispeech_clean100_with_segments")

def extract_segments(textgrid_path, tier_name, sample_rate=16000, hop_length=320):
    tg = TextGrid.fromFile(textgrid_path)
    tier = next(t for t in tg.tiers if t.name.lower() == tier_name)
    segments, labels = [], []
    for interval in tier.intervals:
        mark = interval.mark.strip().lower()
        if mark == "" or (tier_name == "phones" and mark in {"sil", "sp", "spn"}):
            continue
        start = int(float(interval.minTime) * sample_rate / hop_length)
        end = int(float(interval.maxTime) * sample_rate / hop_length) - 1
        segments.append((start, end))
        labels.append(mark)
    return segments, labels

def enrich_dataset_with_segments(dataset_path, textgrid_dir):
    raw_dataset = load_from_disk(dataset_path)
    updated_rows = []
    all_phones = set()
    all_words = set()

    for i, example in enumerate(raw_dataset):
        tg_file = os.path.join(textgrid_dir, f"{i}.TextGrid")
        if not os.path.exists(tg_file):
            continue
        phone_segments, phone_labels = extract_segments(tg_file, "phones")
        word_segments, word_labels = extract_segments(tg_file, "words")
        all_phones.update(phone_labels)
        all_words.update(word_labels)
        updated_rows.append({
            "audio": example["audio"],
            "text": example["text"],
            "input_values": example.get("input_values", None),
            "phone_segments": phone_segments,
            "phone_labels": phone_labels,
            "word_segments": word_segments,
            "word_labels": word_labels,
        })

    phone2id = {p: i for i, p in enumerate(sorted(all_phones))}
    word2id = {w: i for i, w in enumerate(sorted(all_words))}

    df = pd.DataFrame(updated_rows)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(lambda x: {
        "phone_labels": [phone2id[p] for p in x["phone_labels"]],
        "word_labels": [word2id[w] for w in x["word_labels"]],
    })
    return dataset, phone2id, word2id

# Run all steps
updated_dataset, phone2id, word2id = enrich_dataset_with_segments(DATASET_PATH, TEXTGRID_INPUT_DIR)
updated_dataset.save_to_disk(OUTPUT_DATASET_DIR)

with open(os.path.join(OUTPUT_DATASET_DIR, "phone2id.json"), "w") as f:
    json.dump(phone2id, f)
with open(os.path.join(OUTPUT_DATASET_DIR, "word2id.json"), "w") as f:
    json.dump(word2id, f)

print("Segmented dataset saved to:")
print("OUTPUT_DATASET_DIR)
print("#phones:", len(phone2id), "| #words:", len(word2id))
