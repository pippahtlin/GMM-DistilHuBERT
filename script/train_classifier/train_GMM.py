import sys
sys.executable

import torch, gc
gc.collect()
torch.cuda.empty_cache()

import os
import torch
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModel
from train_classifier import extract_all_layer_features

batch_size = 32
save_every = 10  # write to disk every N batches
save_path = "/scratch/pippalin2/jupyter/GMM-DistilHuBERT/script/train_classifier/"
checkpoint_dir = os.path.join(save_path, "save_pathgmm_checkpoints")
resume_file = os.path.join(checkpoint_dir, "gmm_resume_batch.txt")
buffer_file = os.path.join(checkpoint_dir, "pca_buffer.npy")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(checkpoint_dir, exist_ok=True)

# Load model & processor
model = AutoModel.from_pretrained(
    "/mnt/scratch/pippalin2/jupyter/GMM-DistilHuBERT/checkpoints_distilhubert_asr/final_model"
).to(device)
processor = AutoProcessor.from_pretrained(
    "/mnt/scratch/pippalin2/jupyter/GMM-DistilHuBERT/checkpoints_distilhubert_asr/final_model"
)

# Load scaler & PCA
scaler = joblib.load(os.path.join(checkpoint_dir, "scaler.pkl"))
pca = joblib.load(os.path.join(checkpoint_dir, "pca.pkl"))

# Load dataset
dataset = load_dataset(
    "audiofolder",
    data_dir="/mnt/scratch/pippalin2/jupyter/GMM-DistilHuBERT/data/LibriSpeech/train-clean-100"
)["train"]
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Resume logic
start_batch = 0
if os.path.exists(resume_file):
    with open(resume_file, "r") as f:
        start_batch = int(f.read().strip())
    print(f"Resuming from batch {start_batch}")
else:
    print("Starting from batch 0")

# === Main processing
num_batches = (len(dataset) + batch_size - 1) // batch_size
pca_buffer = []

for batch_idx in tqdm(range(start_batch, num_batches)):
    start = batch_idx * batch_size
    end = min((batch_idx + 1) * batch_size, len(dataset))
    batch = dataset.select(range(start, end))

    try:
        waveforms = [ex["audio"]["array"] for ex in batch]
        inputs = processor(
            waveforms, sampling_rate=16000, return_tensors="pt", padding=True
        ).input_values.to(device)

        with torch.no_grad():
            for waveform in inputs:
                feats = extract_all_layer_features(model, waveform.unsqueeze(0), layers=[0, 1, 2])
                feats = feats.squeeze(0).cpu().numpy()
                feats_scaled = scaler.transform(feats)
                feats_pca = pca.transform(feats_scaled)
                pca_buffer.append(feats_pca)

        if (batch_idx + 1) % save_every == 0 or (batch_idx + 1) == num_batches:
            buffer_array = np.vstack(pca_buffer)
            mode = "ab" if os.path.exists(buffer_file) else "wb"
            with open(buffer_file, mode) as f:
                np.save(f, buffer_array)
            pca_buffer = []

        with open(resume_file, "w") as f:
            f.write(str(batch_idx + 1))

    except Exception as e:
        print(f"Batch {batch_idx} skipped due to error: {e}")
        continue


# Load PCA features and train GMM
print("Loading saved PCA features...")
pca_chunks = []
with open(buffer_file, 'rb') as f:
    while True:
        try:
            pca_chunks.append(np.load(f))
        except Exception:
            break
all_pca_feats = np.vstack(pca_chunks)
print("Combined PCA feature shape:", all_pca_feats.shape)

'''
# Train GMM
print("Training GMM...")
gmm = GaussianMixture(n_components=500, covariance_type="diag", reg_covar=1e-2, max_iter=60, verbose=2)

# Randomly sample 4 million rows without replacement
sample_size = 4_000_000
sample_idx = np.random.choice(all_pca_feats.shape[0], size=sample_size, replace=False)
print(f"Fitting GMM on {sample_size} samples...")
gmm.fit(all_pca_feats[sample_idx])

# Save final model + soft labels
joblib.dump(gmm, os.path.join(save_path, "gmm_model.pkl"))
np.save(os.path.join(save_path, "soft_labels.npy"), gmm.predict_proba(all_pca_feats))
print("GMM model and soft labels saved!")
'''