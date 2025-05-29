import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from datasets import load_from_disk
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import Wav2Vec2Processor, Wav2Vec2Model, HubertModel

# === Config ===
data_dir = "/scratch/pippalin2/jupyter/GMM-DistilHuBERT"
dataset_path = os.path.join(data_dir, "data/hf_librispeech_clean100_with_softlabels")
save_dir = os.path.join(data_dir, "script/train_classifier/GMM_DHuBERT_pretrain/gmm_pretrained_distilhubert")
checkpoint_path = os.path.join(save_dir, "checkpoint.pt")
log_path = os.path.join(save_dir, "training_log.csv")
os.makedirs(save_dir, exist_ok=True)

batch_size = 16
num_epochs = 30
learning_rate = 1e-4
mask_prob = 0.065
mask_length = 10
early_stop_patience = 4
hidden_dim = 768
val_split = 0.05

dataset = load_from_disk(dataset_path)

# === Model ===
base_model = HubertModel.from_pretrained("ntu-spml/distilhubert")
base_model.feature_extractor._freeze_parameters()
output_dim = len(dataset[0]["soft_labels"])
proj = nn.Linear(768 * 3, output_dim).to("cuda")
base_model.to("cuda")

optimizer = torch.optim.Adam(
    list(proj.parameters()) + list(base_model.encoder.parameters()),
    lr=learning_rate
)
criterion = nn.KLDivLoss(reduction="batchmean")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# === Load dataset & soft labels ===
def collate_fn(batch):
    audio_arrays = [item["audio"]["array"] for item in batch]
    sampling_rate = batch[0]["audio"]["sampling_rate"]
    soft_labels_batch = [item["soft_labels"] for item in batch]

    inputs = processor(audio_arrays,
                       sampling_rate=sampling_rate,
                       return_tensors="pt",
                       padding="longest").input_values  # Stay on CPU!

    soft_labels_batch = torch.tensor(soft_labels_batch, dtype=torch.float32)

    return inputs, soft_labels_batch


    
total_len = len(dataset)
val_len = int(total_len * val_split)
train_len = total_len - val_len
train_set, val_set = random_split(dataset, [train_len, val_len])
train_loader = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_fn,pin_memory=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=collate_fn, pin_memory=True, num_workers=4)

metrics_log = []


# === Resume checkpoint if exists ===
start_epoch = 0
best_val_loss = float("inf")
no_improve_epochs = 0

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    base_model.load_state_dict(checkpoint["encoder"])
    proj.load_state_dict(checkpoint["proj_head"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["best_val_loss"]
    no_improve_epochs = checkpoint["no_improve_epochs"]
    print(f"Resumed from epoch {start_epoch}")

if os.path.exists(log_path):
    metrics_log = pd.read_csv(log_path).to_dict("records")


for epoch in range(start_epoch, num_epochs):
    base_model.train()
    proj.train()
    total_loss = 0.0
    log_softmax = nn.LogSoftmax(dim=-1)

    for inputs,soft_labels_batch in tqdm(train_loader, desc=f"[Train] Epoch {epoch+1}"):
        inputs = inputs.to("cuda")  # ✅ Here
        soft_labels_batch = soft_labels_batch.to("cuda")

        with torch.no_grad():
            outputs = base_model(inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            features = torch.cat([
                hidden_states[0],  # CNN output
                hidden_states[1],  # Transformer layer 1
                hidden_states[2]   # Transformer layer 2
            ], dim=-1)

            pooled = features.mean(dim=1)  # shape (B, D)
            logits = proj(pooled)          # shape (B, 500)
            log_probs = log_softmax(logits)
            targets = soft_labels_batch.to("cuda")  # shape (B, 500)
            loss = criterion(log_probs, targets)
            total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # === Validation ===
    base_model.eval()
    proj.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inputs,soft_labels_batch in tqdm(val_loader, desc=f"[Val]   Epoch {epoch+1}"):
            inputs = inputs.to("cuda")  #  ^|^e Here
            soft_labels_batch = soft_labels_batch.to("cuda")
            outputs = base_model(inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            features = torch.cat([
                hidden_states[0],
                hidden_states[1],
                hidden_states[2]
            ], dim=-1)

            pooled = features.mean(dim=1)  # shape (B, D)
            logits = proj(pooled)          # shape (B, 500)
            log_probs = log_softmax(logits)
            targets = soft_labels_batch.to("cuda")  # shape (B, 500)
            loss = criterion(log_probs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    if avg_val_loss + 1e-5 < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1
        print(f"No improvement for {no_improve_epochs} epoch(s)")

    metrics_log.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "best_val_loss": best_val_loss,
        "no_improve_epochs": no_improve_epochs
    })
    pd.DataFrame(metrics_log).to_csv(log_path, index=False)
    print(pd.DataFrame(metrics_log).tail())

    torch.save({
        "encoder": base_model.state_dict(),
        "proj_head": proj.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "no_improve_epochs": no_improve_epochs
    }, checkpoint_path)

    if no_improve_epochs >= early_stop_patience:
        print("Early stopping triggered.")
        break
        

# === Save final encoder for ASR fine-tuning ===
base_model.save_pretrained(save_dir)
print("Final pretrained encoder saved.")
