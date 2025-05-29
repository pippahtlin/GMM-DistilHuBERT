import sys
import torch, gc
gc.collect()
torch.cuda.empty_cache()

import os
import numpy as np
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModel
from sklearn.cluster import MiniBatchKMeans

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_path = "/root/final_model"
dataset_path = "/root/LibriSpeech/train-clean-100"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path).to(device)

# Config
layers = [0, 1, 2]
batch_size = 32
max_batches = 1000
n_clusters = 500  # Or whatever you used in your original PDF

# Load dataset
dataset = load_dataset(
                "audiofolder",
                data_dir=dataset_path,
                split="train")
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def extract_all_layer_features(model, waveform, layers=[0, 1, 2]):
    with torch.no_grad():
        outputs = model(waveform, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # tuple: [0]=embedding, [1]=layer1, etc.
        selected = [hidden_states[i] for i in layers]
        return torch.cat(selected, dim=-1)  # [1, T, D_total]
# Feature extraction
features_list = []

for i, example in enumerate(dataset):
    if i >= max_batches:
        break

    input_values = processor(example["audio"]["array"], sampling_rate=16000, return_tensors="pt").input_values.to(device)
    features = extract_all_layer_features(model, input_values, layers=layers)
    features_np = features.squeeze(0).cpu().numpy()

    features_list.append(features_np)

    if i % 10 == 0:
        print(f"Processed {i} batches")

X = np.concatenate(features_list)
print(f"Total features shape: {X.shape}")

# KMeans clustering
kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=8192, verbose=1)
kmeans.fit(X)
print(f"Final inertia: {kmeans.inertia_}")

# Save model
from joblib import dump
save_path = "/root/kmeans_model_100.pkl"
dump(kmeans, save_path)
print(f"K-Means model saved to {save_path}")
