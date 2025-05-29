# Uncertainty-Aware HuBERT: GMM Clustering for Fine-Grained Audio Representation

This repository contains the code and experiments for **SoftHuBERT**, a project exploring the replacement of K-means clustering with **Gaussian Mixture Models (GMM)** in the pretraining pipeline of **Uncertainty-Aware HuBERT**. The goal is to leverage soft, probabilistic cluster assignments to better model uncertainty in speech representations and improve downstream **automatic speech recognition (ASR)** performance.

## Motivation

HuBERT uses K-means clustering to generate discrete targets for masked prediction during self-supervised pretraining. However, K-means provides hard labels and does not capture uncertainty in feature representations. This project proposes replacing K-means with **GMM-based soft labels** and evaluates the effect on fine-tuning performance for ASR tasks.

## Pipeline Overview

1. **Baseline Pretraining (Optional):**  
   Pretrain DistilHuBERT using standard K-means targets to establish a baseline.

2. **Linear Probing for Layer Selection:**  
   Use a linear classifier to identify the most informative hidden layer in the pretrained model (e.g., for phoneme classification).

3. **GMM Training:**  
   Extract hidden representations from the selected layer across the LibriSpeech `train-clean-100` subset.  
   Fit a GMM to those features to generate soft cluster posteriors.

4. **SoftHuBERT Pretraining:**  
   Retrain HuBERT using the GMM soft labels as targets.  
   Replace the cross-entropy loss with **KL divergence**, and optionally freeze lower layers for efficiency.

5. **ASR Fine-Tuning:**  
   Fine-tune the pretrained model on **LibriSpeech-960h** using a CTC loss for ASR.  
   Evaluate on `test-clean` and `test-other`.

## 📁 Directory Structure

```
soft-hubert/
│
├── data/                # LibriSpeech data (loaded via torchaudio)
├── models/              # Model wrappers for HuBERT, DistilHuBERT, and CTC head
├── gmm/                 # GMM training and label generation
├── probing/             # Linear probing scripts for layer selection
├── pretraining/         # SoftHuBERT training with GMM-based targets
├── finetune_asr/        # ASR fine-tuning scripts with CTC
├── utils/               # Utility functions (data loading, logging, etc.)
└── README.md
```

## Dependencies

- PyTorch
- torchaudio
- Transformers (HuggingFace)
- scikit-learn (for GMM)
- tqdm, numpy, pandas

```bash
pip install -r requirements.txt
```

## Dataset

- **LibriSpeech** (train-clean-100 for GMM, train-960 for ASR fine-tuning)  
  Automatically downloaded using `torchaudio.datasets.LIBRISPEECH`.

## 📈 Results (Coming Soon)

| Model             | Pretraining Target | ASR WER (test-clean) | ASR WER (test-other) |
|------------------|--------------------|-----------------------|----------------------|
| Baseline HuBERT  | K-means (hard)     | ...                   | ...                  |
| Uncertainty-Aware HuBERT       | GMM (soft)         | ...                   | ...                  |

## Citation

If you use this project or codebase, please cite it as:

```
@misc{softhubert2025,
  title={Uncertainty-Aware HuBERT: GMM Clustering for Fine-Grained Audio Representation},
  author={Haitao(Pippa) Lin},
  year={2025},
  note={Project repo: https://github.com/yourname/softhubert}
}
```

## Acknowledgments

This project builds on prior work from:
- [HuBERT (Hsu et al., 2021)](https://arxiv.org/abs/2106.07447)
- [DistilHuBERT (Chang et al., 2022)](https://arxiv.org/abs/2110.01900)
