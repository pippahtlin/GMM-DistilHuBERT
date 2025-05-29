import os
import json
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset, Audio
from transformers import AutoProcessor, AutoModel
import pandas as pd
from tqdm import tqdm
import time
import numpy as np

def normalize_features(feats):
    mean = feats.mean(axis=0, keepdims=True)
    std = feats.std(axis=0, keepdims=True) + 1e-6
    return (feats - mean) / std

class DummyMasking(nn.Module):
    def forward(self, x):
        mask = torch.rand_like(x[:, :, 0]) > 0.15
        return x * mask.unsqueeze(-1), mask

def collate_fn(batch):
    waveforms = []
    indices = []
    for item in batch:
        audio = item["audio"]
        if isinstance(audio, dict) and "array" in audio:
            waveforms.append(audio["array"])
            indices.append(item["id"])  # use our custom field
    texts = ["dummy"] * len(waveforms)
    return waveforms, indices

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_path = "/root/final_model"
    knn_model_path = "/root/kmeans_model_100.pkl"
    dataset_path = "/root/LibriSpeech/train-clean-100"
    checkpoint_path = "checkpoint.pt"
    metrics_log_path = "metrics_log.json"
    metrics_csv_path = "metrics_log.csv"

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to(device)
    knn = joblib.load(knn_model_path)
    projection = nn.Linear(model.config.hidden_size * 3, knn.n_clusters).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(projection.parameters()), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    masker = DummyMasking()

    dataset = load_dataset("audiofolder", data_dir=dataset_path, split="train")
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.add_column("id", list(range(len(dataset))))
    print(f"Loaded {len(dataset)} audio examples")

    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_set, val_set = split["train"], split["test"]

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=4, collate_fn=collate_fn)

    start_epoch, best_val_loss, early_stop_counter, patience = 0, float("inf"), 0, 3
    metrics_log = []

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        projection.load_state_dict(checkpoint["projection_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_loss"]
        early_stop_counter = checkpoint["early_stop_counter"]
        print(f"Resumed from epoch {start_epoch}")

    if os.path.exists(metrics_log_path):
        with open(metrics_log_path, "r") as f:
            metrics_log = json.load(f)

    for epoch in range(start_epoch, 80):
        epoch_start = time.time()
        model.train()
        projection.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1} Training")
        for waveforms, indices in loop:
            if not waveforms:
                continue
            for waveform, idx in zip(waveforms, indices):
                try:
                    input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").input_values.to(device)
                    with torch.cuda.amp.autocast():
                        feats = model(input_values, output_hidden_states=True).hidden_states
                        layers_to_concat = feats[0:3]
                        feats_concat = torch.cat(layers_to_concat, dim=-1)
                        masked_feats, _ = masker(feats_concat)

                        labels = np.load(f"/root/frame_labels_clean100/{idx:05d}.npy")
                        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device).view(-1)

                        logits = projection(masked_feats).view(-1, knn.n_clusters)
                        loss = nn.CrossEntropyLoss()(logits, labels_tensor)

                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    running_loss += loss.item()
                except Exception as e:
                    print(f"Training step skipped for {idx:05d} due to error: {e}")

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1} Validation")
        with torch.no_grad():
            for waveforms, indices in val_loop:
                if not waveforms:
                    continue
                for waveform, idx in zip(waveforms, indices):
                    try:
                        input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").input_values.to(device)
                        feats = model(input_values, output_hidden_states=True).hidden_states
                        layers_to_concat = feats[0:3]
                        feats_concat = torch.cat(layers_to_concat, dim=-1)
                        masked_feats, _ = masker(feats_concat)

                        labels = np.load(f"/root/frame_labels_clean100/{idx:05d}.npy")
                        labels_tensor = torch.tensor(labels, dtype=torch.long).to(device).view(-1)

                        logits = projection(masked_feats).view(-1, knn.n_clusters)
                        val_loss += nn.CrossEntropyLoss()(logits, labels_tensor).item()
                    except Exception as e:
                        print(f"Validation step skipped for {idx:05d} due to error: {e}")

        avg_val_loss = val_loss / len(val_loader)
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time/60:.2f} min")

        metrics_log.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        with open(metrics_log_path, "w") as f:
            json.dump(metrics_log, f, indent=2)
        pd.DataFrame(metrics_log).to_csv(metrics_csv_path, index=False)

        torch.save({
            "model_state": model.state_dict(),
            "projection_state": projection.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_loss": best_val_loss,
            "early_stop_counter": early_stop_counter
        }, checkpoint_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered")
                break

if __name__ == "__main__":
    main()
