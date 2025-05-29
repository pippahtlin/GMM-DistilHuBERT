import os
import torch
import numpy as np
import joblib
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModel

def normalize_features(feats):
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    return (feats - mean) / std

model_path = "/root/final_model"
dataset_path = "/root/LibriSpeech/train-clean-100"
kmeans_model_path = "/root/kmeans_model_100.pkl"
save_dir = "/root/frame_labels_clean100"

os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)
model.eval()
kmeans = joblib.load(kmeans_model_path)

dataset = load_dataset("audiofolder", data_dir=dataset_path, split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

for i, example in enumerate(dataset):
    input_values = processor(example["audio"]["array"], sampling_rate=16000, return_tensors="pt").input_values.to(device)
    with torch.no_grad():
        hidden_states = model(input_values, output_hidden_states=True).hidden_states
        feats = [layer.squeeze(0).cpu().numpy() for layer in hidden_states[0:3]]
        feats_concat = np.concatenate(feats, axis=-1)
        feats_norm = normalize_features(feats_concat)
        labels = kmeans.predict(feats_norm)
    np.save(os.path.join(save_dir, f"{i:05d}.npy"), labels)
    if i % 10 == 0:
        print(f"Saved labels for utterance {i}")
