import os
import torch
import numpy as np
from datasets import load_dataset, load_from_disk
import evaluate
import jiwer
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC,
                          TrainingArguments, Trainer, EarlyStoppingCallback)
import pandas as pd
import torchaudio

transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])

# === Updated for KNN ===
data_dir = "/root"
pretrained_model_dir = os.path.join(data_dir, "Kmeans-distilhubert-best-encoder")
output_dir = os.path.join(data_dir, "root/kmeans_final_asr_output")
metrics_csv_path = os.path.join(output_dir, "KNN_final_metrics.csv")

batch_size = 8
num_epochs = 10
os.makedirs(output_dir, exist_ok=True)

processor = Wav2Vec2Processor.from_pretrained(pretrained_model_dir)
tokenizer = processor.tokenizer

train_dataset = load_from_disk(os.path.join(data_dir, "root/hf_librispeech_clean100_preprocessed"))["train"]

print("Loading pretrained KNN-DistilHuBERT encoder...")
model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_dir, vocab_size=len(tokenizer))

training_args = TrainingArguments(
    output_dir=output_dir,
    group_by_length=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=num_epochs,
    fp16=True,
    save_total_limit=3,
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=100,
    report_to=["none"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False
)

# === Metrics ===
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
metrics_log = []

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred_str = [transform(p) for p in tokenizer.batch_decode(pred_ids)]
    label_str = [transform(l) for l in tokenizer.batch_decode(pred.label_ids, group_tokens=False)]

    result = {
        "wer": wer_metric.compute(predictions=pred_str, references=label_str),
        "cer": cer_metric.compute(predictions=pred_str, references=label_str),
    }

    # Log training loss if available
    if hasattr(pred, "metrics") and "loss" in pred.metrics:
        result["loss"] = pred.metrics["loss"]

    metrics_log.append(result)
    pd.DataFrame(metrics_log).to_csv(metrics_csv_path, index=False)
    return result


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Starting fine-tuning...")
trainer.train(resume_from_checkpoint=True)
trainer.save_model(output_dir)
print("Fine-tuning complete. Model saved to:", output_dir)

# === Final Evaluation ===
print("Running final evaluation on test-clean...")
eval_result = trainer.evaluate()
final_result = {"wer": eval_result["eval_wer"], "cer": eval_result["eval_cer"]}
metrics_log.append(final_result)
pd.DataFrame(metrics_log).to_csv(metrics_csv_path, index=False)
print("Final WER/CER saved to:", metrics_csv_path)