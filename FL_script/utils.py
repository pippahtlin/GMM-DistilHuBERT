from typing import List, Dict
import torch
import copy
import numpy as np
import os
import random
from transformers import Trainer, TrainingArguments
from opacus.accountants import RDPAccountant
import jiwer

# Set global seeds for reproducibility
GLOBAL_SEED = 217
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#### Regular FL ####

def split_into_clients_nonuniform(dataset, num_clients, min_frac, max_frac, seed=None):
    if seed is not None:
        np.random.seed(seed)

    size = len(dataset)
    proportions = np.random.uniform(min_frac, max_frac, size=num_clients)
    proportions = proportions / proportions.sum()
    sizes = (proportions * size).astype(int)
    diff = size - sizes.sum()
    sizes[0] += diff

    indices = np.arange(size)
    np.random.shuffle(indices)

    client_datasets = []
    start = 0
    for s in sizes:
        end = start + s
        client_datasets.append(dataset.select(indices[start:end].tolist()))
        start = end

    return client_datasets

class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding=True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        batch = self.processor.feature_extractor.pad(input_features, padding=self.padding, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, padding=self.padding, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

def local_finetune(model, dataset, processor, collator, compute_metrics, output_dir):
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        eval_strategy="no",
        num_train_epochs=2,
        eval_steps=400,
        logging_steps=10,
        save_steps=500,
        learning_rate=1e-4,
        fp16=True,
        resume_from_checkpoint=False,
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=processor,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    train_output = trainer.train()
    metrics = {
        "train_loss": train_output.training_loss,
        "train_steps": trainer.state.global_step,
    }
    return model.state_dict(), metrics

def fed_avg(state_dicts: List[Dict]):
    avg_dict = copy.deepcopy(state_dicts[0])
    for key in avg_dict:
        for i in range(1, len(state_dicts)):
            avg_dict[key] += state_dicts[i][key]
        avg_dict[key] = avg_dict[key] / len(state_dicts)
    return avg_dict

processor = None
transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
])
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)
    pred_str = processor.batch_decode(pred_ids)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, group_tokens=False)
    pred_str = [transform(p) for p in pred_str]
    label_str = [transform(l) for l in label_str]
    wer = jiwer.wer(label_str, pred_str)
    cer = jiwer.cer(label_str, pred_str)
    ser = sum(p != l for p, l in zip(pred_str, label_str)) / len(label_str)
    return {
        "wer": wer,
        "cer": cer,
        "ser": ser,
    }

def evaluate_global_model(global_model, eval_dataset, processor, data_collator, model_save_path):
    eval_args = TrainingArguments(
        output_dir=model_save_path,
        per_device_eval_batch_size=4,
        report_to="none",
        do_train=False,
        do_eval=True,
        dataloader_drop_last=False,
    )
    eval_trainer = Trainer(
        model=global_model,
        args=eval_args,
        tokenizer=processor.tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        eval_dataset=eval_dataset
    )
    results = eval_trainer.evaluate()
    print(f"Eval after round: WER={results['eval_wer']:.4f}, CER={results['eval_cer']:.4f}, SER={results['eval_ser']:.4f}")
    return results

def flat_clip(delta_k, S):
    norm = torch.norm(delta_k)
    if norm > S:
        delta_k = delta_k * (S / norm)
    return delta_k

def dp_aggregate_f_c(updates, weights, q, Wmin):
    W = sum(weights)
    denom = max(q * Wmin, W)
    agg = {}
    for key in updates[0].keys():
        agg[key] = sum([w * u[key] for u, w in zip(updates, weights)]) / denom
    return agg

def compute_sigma(z, S, q, Wmin):
    return z * (2 * S) / (q * Wmin)

def train_dp_fedavg(global_model, z, S, label, q, Wmin, rounds, processor, data_collator, compute_metrics, eval_dataset, client_datasets, model_save_path, delta=0.05, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    model = copy.deepcopy(global_model)
    accountant = RDPAccountant()
    accs = []
    epsilons = []
    train_losses = []
    sigma = compute_sigma(z, S, q, Wmin)
    print(f"[{label}] Using S={S}, z={z}, sigma={sigma:.4f}")

    os.makedirs(model_save_path, exist_ok=True)

    for rnd in range(rounds):
        selected = random.sample(client_datasets, int(q * len(client_datasets)))
        updates = []
        weights = []
        round_losses = []

        for client_data in selected:
            local_model = copy.deepcopy(model)
            state_before = copy.deepcopy(local_model.state_dict())
            state_after, metrics = local_finetune(local_model, client_data, processor, data_collator, compute_metrics, model_save_path)
            round_losses.append(metrics["train_loss"])
            update_delta = {k: state_after[k] - state_before[k] for k in state_before}
            clipped_delta = {k: flat_clip(v, S) for k, v in update_delta.items()}
            updates.append(clipped_delta)
            weights.append(1.0)

        avg_update = dp_aggregate_f_c(updates, weights, q, Wmin)
        noise = {k: torch.normal(0, sigma, size=v.size()).to(v.device) for k, v in avg_update.items()}
        noisy_update = {k: avg_update[k] + noise[k] for k in avg_update}

        new_state = model.state_dict()
        for k in new_state:
            new_state[k] += noisy_update[k]
        model.load_state_dict(new_state)

        accountant.step(noise_multiplier=z, sample_rate=q)
        epsilon = accountant.get_epsilon(delta)
        epsilons.append(epsilon)

        metrics = evaluate_global_model(model, eval_dataset, processor, data_collator, model_save_path)
        accs.append(metrics['eval_wer'])
        avg_loss = sum(round_losses) / len(round_losses)
        train_losses.append(avg_loss)

        print(f"Round {rnd + 1}, σ={sigma:.4f}, ε={epsilon:.4f}, WER={metrics['eval_wer']:.3f}, Loss={avg_loss:.4f}")

        model.save_pretrained(os.path.join(model_save_path, f"round_{rnd + 1}_model"))

    return {
        "label": label,
        "S": S,
        "sigma": sigma,
        "z": z,
        "wer_curve": accs,
        "epsilon_curve": epsilons,
        "train_loss_curve": train_losses
    }
