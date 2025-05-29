import os
import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model

# === Config ===
data_dir = "/scratch/pippalin2/jupyter/GMM-DistilHuBERT"
dataset_path = os.path.join(data_dir, "data/hf_librispeech_clean100")
label_path = os.path.join(data_dir, "script/train_classifier/soft_labels_clean100.npy")
batch_size = 8
num_epochs = 5
learning_rate = 1e-4
mask_prob = 0.065
mask_length = 10
hidden_dim = 768  # for DistilHuBERT
early_stop_patience = 5

# === Load data ===
dataset = load_from_disk(dataset_path)["train"]
soft_labels = np.load(label_path)  # shape: [total_frames, num_clusters]
label_index = 0  # global pointer

# === Model setup ===
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
base_model = Wav2Vec2Model.from_pretrained("ntu-spml/distilhubert")

# Freeze CNN layers
base_model.feature_extractor._freeze_parameters()

# Projection head
proj = nn.Linear(hidden_dim, soft_labels.shape[1]).to("cuda")
optimizer = torch.optim.Adam(proj.parameters(), lr=learning_rate)
criterion = nn.KLDivLoss(reduction="batchmean")

# Time masking
def apply_time_mask(x, mask_prob=0.065, mask_length=10):
    B, T, D = x.shape
    mask = torch.ones((B, T), dtype=torch.bool, device=x.device)
    for b in range(B):
        num_spans = int(mask_prob * T / mask_length)
        for _ in range(num_spans):
            start = np.random.randint(0, max(T - mask_length, 1))
            mask[b, start:start + mask_length] = False
    return mask

# === Training ===
train_loader = DataLoader(dataset, batch_size=batch_size)

for epoch in range(num_epochs):
    proj.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        # === Preprocess audio ===
        inputs = processor(batch["audio"]["array"],
                           sampling_rate=batch["audio"]["sampling_rate"][0],
                           return_tensors="pt", padding=True).input_values.to("cuda")

        with torch.no_grad():
            features = base_model(inputs).last_hidden_state  # [B, T, D]

        mask = apply_time_mask(features, mask_prob, mask_length)
        masked_features = features[~mask]  # [N, D]

        # === Match soft labels ===
        global label_index
        num_masked = masked_features.size(0)
        if label_index + num_masked > soft_labels.shape[0]:
            print("Ran out of soft labels!")
            break

        targets = torch.tensor(soft_labels[label_index:label_index + num_masked], dtype=torch.float32).to("cuda")
        label_index += num_masked

        # === Loss and optimize ===
        logits = proj(masked_features)
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = criterion(log_probs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")
