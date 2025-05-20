# DL_Assignment_03 – Transliteration using Seq2Seq and Attention

This repository contains code and experiments for **Assignment 3** of Deep Learning, focusing on building a character-level transliteration system using Recurrent Neural Networks (RNNs) and attention mechanisms.

## Problem Statement

Given a sequence of characters in a romanized script (e.g., `ghar`), the goal is to train a model that outputs its transliteration in Devanagari script (e.g., `घर`). This is a sequence-to-sequence learning task, similar to neural machine translation at the character level.

The experiments are conducted using the [Dakshina dataset](https://github.com/google-research-datasets/dakshina).

## Wandb report link: https://api.wandb.ai/links/ee22s084-indian-institute-of-technology-madras/k2o17ca9

## Model Architecture

Two main models are implemented:

### Vanilla Seq2Seq Model (No Attention)
- Input Embedding Layer
- Encoder: RNN / LSTM / GRU (configurable)
- Decoder: RNN / LSTM / GRU (configurable)
- Teacher Forcing
- Beam Search (optional)

### Seq2Seq with Attention
- Same as above, but with an attention layer between encoder and decoder.


## Hyperparameter Tuning

Bayesian Optimization was used via [Weights & Biases sweeps](https://wandb.ai). Key hyperparameters:
- `embedding_size`: 128, 256
- `hidden_size`: 128, 256
- `encoder_num_layers`: 1, 2, 3
- `decoder_num_layers`: 1, 2
- `encoder_name`, `decoder_name`: RNN / GRU / LSTM
- `dropout_p`: 0.2, 0.3
- `learning_rate`: 0.001, 0.003
- `teacher_forcing_p`: 0.5, 0.8
- `encoder_bidirectional`: True / False
- `beam_size`: 1

---

##  Results

### Best Config (Vanilla Seq2Seq)
```yaml
encoder_name: GRU
decoder_name: GRU
encoder_num_layers: 2
decoder_num_layers: 2
embedding_size: 128
hidden_size: 256
dropout_p: 0.3
teacher_forcing_p: 0.8
beam_size: 1


Train Accuracy: 87.5%
Validation Accuracy: 76.9%
Test Accuracy: 20.5%

Best Config (Attention-Based)
yaml
Copy
Edit
encoder_name: GRU
decoder_name: RNN
encoder_num_layers: 1
decoder_num_layers: 1
embedding_size: 256
hidden_size: 128
dropout_p: 0.3
teacher_forcing_p: 0.5
beam_size: 1
apply_attention: True



├── data/                     # Preprocessed Dakshina dataset
├── models/                   # Model definitions (Vanilla, Attention)
├── trainer.py                # Training loop and evaluation logic
├── sweeps/                   # W&B sweep config files
├── predictions_vanilla/      # Outputs from best vanilla model
├── predictions_attention/    # Outputs from attention-based model
└── README.md                 # This file

