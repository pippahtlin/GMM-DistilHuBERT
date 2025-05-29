import os
import torch
import numpy as np
from datasets import load_dataset, load_from_disk
import evaluate
import jiwer
from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC,
                          TrainingArguments, Trainer, AutoModel, Trainer)
import torchaudio
import pandas as pd

transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip()
])

data_dir = "/scratch/pippalin2/jupyter/GMM-DistilHuBERT"
pretrained_model_dir = os.path.join(data_dir, "script/train_classifier/GMM_DHuBERT_pretrain/gmm_pretrained_distilhubert")  
output_dir = os.path.join(data_dir, "script/train_classifier/GMM_final_finetune/final_asr_output")
metrics_csv_path = os.path.join(output_dir, "GMM_final_metrics.csv")
batch_size = 8
num_epochs = 8
os.makedirs(output_dir, exist_ok=True)

processor = Wav2Vec2Processor.from_pretrained(pretrained_model_dir)
tokenizer = processor.tokenizer

dataset = load_from_disk(os.path.join(data_dir, "data/hf_librispeech_clean100_preprocessed"))["train"]
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

test_dataset = load_from_disk(os.path.join(data_dir, "data/hf_test_clean_preprocessed"))["train"]

from typing import List, Dict, Union
import torch

class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding: Union[bool, str] = True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

        # Padding
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt"
        )
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt"
        )

        # Replace padding token IDs with -100 for CTC loss to ignore
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

print("Loading pretrained GMM-DistilHuBERT encoder...")
# Load pretrained GMM model
model = Wav2Vec2ForCTC.from_pretrained(pretrained_model_dir,vocab_size=len(tokenizer))

training_args = TrainingArguments(
    output_dir=output_dir,
    group_by_length=True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    num_train_epochs=num_epochs,
    fp16=True,
    save_total_limit=2,
    logging_steps=100,
    logging_dir=os.path.join(output_dir, "logs"),
    report_to=["none"]
)


# Metric
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
metrics_log = []

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred_str = [transform(p) for p in tokenizer.batch_decode(pred_ids)]
    label_str = [transform(l) for l in tokenizer.batch_decode(pred.label_ids, group_tokens=False)]

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    result = {"wer": wer, "cer": cer}
    print(f"📊 Eval — WER: {wer:.4f} | CER: {cer:.4f}")

    metrics_log.append(result)
    pd.DataFrame(metrics_log).to_csv(metrics_csv_path, index=False)
    return result

# test
train_dataset = train_dataset.select(range(len(train_dataset)//15))


# Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting fine-tuning...")
trainer.train() #resume_from_checkpoint=True)
trainer.save_model(output_dir)
print("Fine-tuning complete. Model saved to:", output_dir)


# Final Eval
final_results = trainer.evaluate(eval_dataset=test_dataset)

# Log or save the results
print("Final WER:", final_results["eval_wer"])
print("Final CER:", final_results["eval_cer"])
metrics_log.append({
    "dataset": "test-clean",
    "wer": final_results["eval_wer"],
    "cer": final_results["eval_cer"]
})
pd.DataFrame(metrics_log).to_csv(metrics_csv_path, index=False)
