import sys
sys.executable

import torch, gc
gc.collect()
torch.cuda.empty_cache()

import os
import numpy as np
import joblib
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# === Configs ===
checkpoint_dir = "/scratch/pippalin2/jupyter/GMM-DistilHuBERT/script/train_classifier/save_pathgmm_checkpoints"
save_path = "/scratch/pippalin2/jupyter/GMM-DistilHuBERT/script/train_classifier"
buffer_file = os.path.join(checkpoint_dir, "pca_buffer.npy")
gmm_model_path = os.path.join(save_path, "gmm_model_subset4M.pkl")
soft_labels_path = os.path.join(save_path, "soft_labels_all.npy")

# === Load saved PCA features ===
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

# === Randomly sample 4M features for GMM training ===
sample_size = 1_000_000
print(f"Sampling {sample_size} features for GMM training...")
sample_idx = np.random.choice(all_pca_feats.shape[0], size=sample_size, replace=False)
sample_feats = all_pca_feats[sample_idx]

# === Train GMM ===
print("Training GMM on sampled subset...")
gmm = GaussianMixture(n_components=500, covariance_type="diag", reg_covar=1e-2, max_iter=60, verbose=2, init_params="random")
gmm.fit(sample_feats)

# === Save trained GMM ===
joblib.dump(gmm, gmm_model_path)
print(f"Saved GMM model to {gmm_model_path}")


# === Generate soft labels for all 23M features in batches ===
print("Generating soft labels for all features...")

batch_size = 1_000_000
soft_chunks = []

for i in tqdm(range(0, all_pca_feats.shape[0], batch_size)):
    batch = all_pca_feats[i:i+batch_size]
    soft = gmm.predict_proba(batch)
    soft_chunks.append(soft)

soft_labels = np.vstack(soft_chunks)
np.save(soft_labels_path, soft_labels)
print(f"Saved soft labels to {soft_labels_path}")
