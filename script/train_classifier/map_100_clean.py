# This is how you want to map first so you can rerun (the final finetuning) as much as possible 
# But here it is using huggingface model, you should change it to your own model tomorrow.
from datasets import load_from_disk, DatasetDict
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor
import jiwer
import torchaudio
import os
import pandas as pd

# ==== Paths ====
input_path = "/scratch/pippalin2/jupyter/GMM-DistilHuBERT/data/hf_librispeech_clean100"
output_path = "/scratch/pippalin2/jupyter/GMM-DistilHuBERT/data/hf_librispeech_clean100_preprocessed"

# ==== Load raw dataset ====
ds = load_from_disk(input_path)

# ==== Load tokenizer/processor ====
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# ==== Text cleaning ====
transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])

def clean_text(batch):
    batch["text"] = transform(batch["text"])
    return batch

# ==== Feature extraction ====
def extract_features(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

# ==== Apply transforms ====
ds = ds.map(clean_text)
ds = ds.map(extract_features, remove_columns=ds.column_names)

# ==== Save preprocessed dataset ====
DatasetDict({"train": ds}).save_to_disk(output_path)
print(f"✅ Preprocessed dataset saved to: {output_path}")
