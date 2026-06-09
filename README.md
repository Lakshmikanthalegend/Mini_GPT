# TinyGPT - Simple Transformer Language Model

## Overview
This project implements a small GPT-style transformer model using PyTorch.  
It is trained on a custom text corpus to generate new sentences using next-word prediction.

---

## Features
- Custom tokenizer (word-level)
- Transformer architecture with attention
- Position + token embeddings
- Text generation using sampling
- Minimal GPT implementation for learning

---

## Model Architecture
- Embedding Layer
- Positional Encoding
- Multi-head Attention (Transformer Blocks)
- Feed Forward Layers
- Layer Normalization
- Linear Output Layer

---

## Dataset
A small custom corpus of sentences related to daily language.

Example:
- "diwali brings lights and sweets"
- "the tea is very hot"
- "RCB won the IPL match"

---

## Hyperparameters
- Block size: 6
- Embedding dimension: 32
- Heads: 2
- Layers: 2
- Epochs: 1500

---

## Training
Model learns using next-token prediction:

Input:  [word1, word2, word3]  
Target: [word2, word3, word4]

Loss Function:
- Cross Entropy Loss

Optimizer:
- AdamW

---

## Text Generation
The model generates text word-by-word by:
1. Taking last context
2. Predicting next word probabilities
3. Sampling next word
4. Appending to sequence

---

## Example Output
Input: "diwali"  
Output:
diwali brings lights and sweets <END>

(Note: Output varies due to randomness)

---

## How to Run

```bash
python demo.py