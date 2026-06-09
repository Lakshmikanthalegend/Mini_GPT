# Mini GPT

A minimal GPT-style language model built from scratch in PyTorch. The project implements the core transformer building blocks (self-attention, multi-head attention, feed-forward network, transformer block) and a small word-level demo that trains on a tiny corpus.

## Project Structure

```
MiniProj/
├── LLM_Mini/
│   ├── demo.py                # Tokenization + hyperparameters + training entry point
│   └── transformer_blocks.py  # SelfAttentionHead, MultiHeadAttention, FeedForward, Block
├── .gitignore
└── README.md
```

## Requirements

- Python 3.10+ (3.11/3.12 recommended)
- PyTorch 2.x
- (Optional) NVIDIA GPU + CUDA for faster training

---

## 1. Clone the Repository

```powershell
git clone https://github.com/Lakshmikanthalegend/Mini_GPT.git
cd Mini_GPT
```

## 2. Create and Activate a Virtual Environment

### Windows (PowerShell)
```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

If activation is blocked, run once in an elevated PowerShell:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### Windows (cmd)
```bat
python -m venv myenv
myenv\Scripts\activate.bat
```

### macOS / Linux
```bash
python3 -m venv myenv
source myenv/bin/activate
```

## 3. Install Dependencies

```powershell
python -m pip install --upgrade pip
pip install torch
```

For a CUDA build (example for CUDA 12.1), use the official selector at https://pytorch.org/get-started/locally/ — e.g.:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 4. Run the Demo

```powershell
cd LLM_Mini
python demo.py
```

You should see your Torch version, CUDA availability, the tiny corpus, the vocabulary, and the tokenized data printed to the console.

---

## Common Git Workflow

Make changes locally and push them back to GitHub:

```powershell
# See what changed
git status

# Stage changes
git add .

# Commit with a message
git commit -m "Describe your change"

# Push to GitHub (first push from a new branch uses -u)
git push -u origin main
```

Pull the latest changes from GitHub:
```powershell
git pull origin main
```

Create a feature branch:
```powershell
git checkout -b feature/my-change
# ...edit files...
git add .
git commit -m "Add my change"
git push -u origin feature/my-change
```

---

## Reusing This Project from Scratch (Quick Reference)

```powershell
# 1. Clone
git clone https://github.com/Lakshmikanthalegend/Mini_GPT.git
cd Mini_GPT

# 2. Virtual env
python -m venv myenv
.\myenv\Scripts\Activate.ps1

# 3. Install deps
pip install --upgrade pip
pip install torch

# 4. Run
cd LLM_Mini
python demo.py
```

---

## Notes

- The `myenv/` virtual environment folder is intentionally excluded via `.gitignore`. Each user should create their own locally.
- The current `demo.py` covers tokenization, vocabulary, and hyperparameter setup. Extend it with a training loop that uses `Block` from `transformer_blocks.py` to train and generate text.
