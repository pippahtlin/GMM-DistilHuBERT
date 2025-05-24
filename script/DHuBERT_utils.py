import torch
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.mixture import GaussianMixture
import jiwer
from typing import List, Dict, Union


def extract_layer_representations(model, processor, dataset, layer_index, batch_size=8):
    def get_features_batch(batch):
        input_values = batch['input_values']
        if isinstance(input_values[0], list):  # convert list-of-lists to tensors
            input_values = [torch.tensor(x) for x in input_values]

        # Pad and create attention mask
        input_tensor = torch.nn.utils.rnn.pad_sequence(input_values, batch_first=True)
        attention_mask = torch.tensor([
            [1]*len(x) + [0]*(input_tensor.shape[1] - len(x)) for x in input_values
        ])

        input_tensor = input_tensor.to('cuda')
        attention_mask = attention_mask.to('cuda')

        with torch.no_grad():
            outputs = model.hubert(input_tensor, attention_mask=attention_mask, output_hidden_states=True)
        layer_outputs = outputs.hidden_states[layer_index].cpu().numpy()

        return {'layer_output': [layer_outputs[i] for i in range(len(input_values))]}

    return dataset.map(
        get_features_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=[],  # keep empty or list of columns to remove
        desc=f"Extracting layer {layer_index} representations"
    )


def pool_segment_features(layer_output, segments):
    """Averages hidden states over specified time-frame segments."""
    return [layer_output[start:end + 1].mean(axis=0) for start, end in segments]


def prepare_onehot_labels(labels):
    encoder = OneHotEncoder(sparse=False)
    return encoder.fit_transform(np.array(labels).reshape(-1, 1))


def get_cca_similarity(x, y, epsilon=1e-10):
    from scipy import linalg

    x -= x.mean(axis=0)
    y -= y.mean(axis=0)

    Sigma_xx = np.dot(x.T, x) / x.shape[0] + epsilon * np.eye(x.shape[1])
    Sigma_yy = np.dot(y.T, y) / y.shape[0] + epsilon * np.eye(y.shape[1])
    Sigma_xy = np.dot(x.T, y) / x.shape[0]

    inv_Sigma_xx = linalg.inv(linalg.cholesky(Sigma_xx, lower=True))
    inv_Sigma_yy = linalg.inv(linalg.cholesky(Sigma_yy, lower=True))

    T = np.dot(np.dot(inv_Sigma_xx.T, Sigma_xy), inv_Sigma_yy)
    U, s, Vh = linalg.svd(T)

    x_proj = np.dot(x, np.dot(inv_Sigma_xx.T, U))
    x_weights = np.sum(x_proj**2, axis=0)
    x_weights /= x_weights.sum()

    y_proj = np.dot(y, np.dot(inv_Sigma_yy.T, Vh.T))
    y_weights = np.sum(y_proj**2, axis=0)
    y_weights /= y_weights.sum()

    return {
        "cca_coef": s,
        "x_weights": x_weights,
        "y_weights": y_weights,
    }


def compute_pwcca_similarity(representations, label_vectors, epsilon=1e-4):
    cca_result = get_cca_similarity(x=representations, y=label_vectors, epsilon=epsilon)
    pwcca_score = np.sum(cca_result["cca_coef"] * cca_result["x_weights"])
    return pwcca_score


def train_gmm_on_features(features, num_clusters=50):
    gmm = GaussianMixture(n_components=num_clusters, covariance_type='diag', max_iter=100)
    gmm.fit(features)
    return gmm


def prepare_example(example, processor):
    audio = example["audio"]
    example["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        example["labels"] = processor(example["text"]).input_ids
    return example


class DataCollatorCTCWithPadding:
    def __init__(self, processor, padding: Union[bool, str] = True):
        self.processor = processor
        self.padding = padding

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]

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

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch


def compute_metrics(pred, processor):
    transform = jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])

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


def clear_gpu_cache():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
