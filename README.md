# AG News Text Classifier

This project implements a simple **RNN/LSTM-based text classification model** to classify news articles from the AG News dataset.

---

## 📚 Installation

First, clone the repository and install the required Python packages:

```bash
git clone <your-repo-url>
cd AGNews_Classifier
pip install -r requirements.txt

## Features

Tokenization using BERT tokenizer.

Supports RNN and LSTM models with optional bidirectionality.

Handles padding and truncation of sequences.

Trains and evaluates on the AG News dataset with 4 categories:

1.World

2.Sports

3.Business

4.Sci/Tech

## requirements

torch
torchvision
torchtext
transformers
datasets
tqdm
matplotlib
