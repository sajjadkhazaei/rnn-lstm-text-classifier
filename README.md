📰 AG News Text Classifier

This project implements a deep learning–based text classification model using RNN / LSTM (or GRU) to classify news articles from the AG News dataset into four categories.

The goal of this project is to demonstrate a complete NLP pipeline, from dataset loading and preprocessing to model training, evaluation, and inference — all inside a single Jupyter Notebook.

📌 Project Overview

Dataset: AG News

Task: Multi-class text classification

Classes:

World 🌍

Sports 🏅

Business 💼

Sci/Tech 🔬

Tokenizer: BERT Tokenizer (bert-base-uncased)

Models Supported:

RNN

LSTM

GRU

Framework: PyTorch

📂 Notebook Structure
1️⃣ Imports & Setup

Install required libraries (datasets, transformers, torchmetrics)

Import Python and PyTorch modules

Select device (cuda / cpu)

2️⃣ Dataset & Tokenizer

Load the AG News dataset using datasets.load_dataset

Define BertTokenizerFast

Tokenization, padding, and truncation

Set PyTorch format (input_ids, attention_mask, label)

Create DataLoaders for training and testing

3️⃣ Utils

Define AverageMeter to compute average loss

Additional helper utilities (e.g., accuracy tracking)

4️⃣ Model Definition

Define RNNModel class supporting RNN / LSTM / GRU

Architecture:

Embedding layer

Recurrent layer

Fully connected classification layer

5️⃣ Training Function

train_one_epoch function

Batch-wise training loop

Loss and accuracy computation

6️⃣ Validation Function

Validation loop without backpropagation

Computes validation loss and accuracy

7️⃣ Training Loop

Loop over multiple epochs

Save the best model using torch.save

Store training and validation metrics for visualization

8️⃣ Plot Results

Plot:

Training vs Validation Loss

Training vs Validation Accuracy

9️⃣ Inference

Perform predictions on:

Custom input sentences

Batches from the test set

🚀 Features

Tokenization using BERT tokenizer

Supports RNN, LSTM, and GRU

Optional bidirectional RNNs

Proper handling of padding and truncation

End-to-end training, evaluation, and inference

Clean and educational notebook-style implementation

📚 Installation

Clone the repository and install dependencies:

git clone <your-repo-url>
cd rnn-lstm-text-classifier
pip install -r requirements.txt

📦 Requirements
torch
torchvision
torchtext
transformers
datasets
torchmetrics
tqdm
matplotlib

📊 Example Inference
Text: "Apple releases a new iPhone model"
Prediction: Business

🎯 Purpose of This Project

This project is designed to:

Practice NLP with deep learning

Understand RNN/LSTM behavior on text

Build a portfolio-ready project

Prepare for ML / DL interviews and real-world tasks
