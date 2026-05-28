# Sentiment Classification using Transformer and BERT Models

## Overview

This repository contains the implementation of Transformer-based sentiment classification models using PyTorch on the SST-2 dataset. The project was developed as part of the **AI61002: Deep Learning Foundations and Applications** course assignment.

The project focuses on designing custom Transformer encoder architectures, comparing them with PyTorch’s built-in Transformer modules, and applying transfer learning using a pre-trained BERT encoder for sentiment classification.

---

## Project Objectives

* Load and preprocess the SST-2 sentiment classification dataset
* Implement custom Transformer encoder layers
* Build Transformer-based text classification models
* Compare custom Transformer and built-in TransformerEncoderLayer implementations
* Apply transfer learning using a pre-trained BERT encoder
* Train and evaluate all models on sentiment classification tasks
* Analyze performance using classification metrics

---

## Technologies Used

* Python
* PyTorch
* TorchText
* Hugging Face Transformers
* NumPy
* Matplotlib
* Scikit-learn
* Jupyter Notebook

---

## Dataset

The project uses the **SST-2 (Stanford Sentiment Treebank 2)** dataset for binary sentiment classification.

### Classes

* Positive
* Negative

The dataset was loaded using `torchtext.datasets`.

---

## Text Preprocessing

The following preprocessing steps were implemented:

* Stopword removal
* Punctuation removal
* Tokenization using BERT Tokenizer
* Vocabulary creation using `torchtext.vocab`
* Text-to-index conversion
* One-hot encoding of labels

---

## Model Architectures

Three different architectures were implemented and compared.

---

## Model 1 — Custom Transformer Encoder

The custom Transformer model was built using:

* `nn.MultiheadAttention`
* Residual connections
* Layer normalization
* Position-wise feedforward network
* Sinusoidal positional encoding
* Dropout regularization

### Features

* Embedding layer using `nn.Embedding`
* Custom encoder layer implementation
* Multi-head self-attention mechanism
* Classification layer after encoder output

---

## Model 2 — Built-in Transformer Encoder

The custom encoder was replaced with:

```python id="zv8pzy"
nn.TransformerEncoderLayer()
```

### Features

* Built-in Transformer architecture
* Same hyperparameters as custom model
* Faster and cleaner implementation
* Comparable performance evaluation

---

## Model 3 — BERT Transfer Learning

Transfer learning was performed using:

```python id="5y3z0y"
bert-base-uncased
```

### Features

* Frozen pre-trained BERT encoder
* Linear classification head
* Contextual embedding extraction
* Improved sentiment classification performance

---

## Training Details

### Common Hyperparameters

| Parameter     | Value            |
| ------------- | ---------------- |
| Optimizer     | Adam             |
| Learning Rate | 1e-3             |
| Loss Function | CrossEntropyLoss |
| Batch Size    | 32               |
| Epochs        | 5                |

### Transformer Hyperparameters

| Parameter                  | Value |
| -------------------------- | ----- |
| Hidden Dimension (d_model) | 16    |
| Attention Heads            | 4     |
| Encoder Layers             | 1     |
| Dropout                    | 0.5   |
| Max Positional Length      | 5000  |

---

## Evaluation Metrics

All models were evaluated using:

* Accuracy
* Precision
* Recall

Training and validation losses were also visualized across epochs.

---

## Project Structure

```bash id="fql4nq"
├── assignment_4.ipynb          # Jupyter Notebook implementation
├── assignment_4.py             # Python script implementation
├── models/                     # Saved model checkpoints
├── plots/                      # Loss curves and visualizations
├── results/                    # Evaluation outputs
├── README.md                   # Project documentation
```

---

## How to Run

### Clone the Repository

```bash id="i5fxvk"
git clone <your-repository-link>
cd <repository-folder>
```

### Install Dependencies

```bash id="8t0q1x"
pip install torch torchtext transformers numpy matplotlib scikit-learn
```

### Run the Notebook

```bash id="s8rgmg"
jupyter notebook
```

Open:

```bash id="s3bjlwm"
assignment_4.ipynb
```

---

## Results

The Transformer-based models successfully learned contextual text representations for sentiment classification.

Key observations:

* Self-attention improved sequence understanding
* Built-in Transformer layers simplified implementation
* BERT transfer learning achieved the best performance
* Positional encoding improved sequential information modeling
* Dropout regularization reduced overfitting

---

## Learning Outcomes

This project helped in understanding:

* Transformer architectures
* Multi-head self-attention
* Positional encoding
* Residual connections
* Sequence modeling
* Transfer learning
* BERT-based NLP models
* Sentiment classification pipelines

---

## Assignment Reference

This implementation follows the requirements specified in the AI61002 Assignment 4 guidelines.

---

## Author

Mahesh Kusireddy

AI61002 - Deep Learning Foundations and Applications
Indian Institute of Technology Kharagpur
