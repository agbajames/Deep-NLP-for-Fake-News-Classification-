# Fake News Detection with XLNet

Fine-tuning a transformer model to classify real vs fake news articles using the FakeNewsNet dataset.

---

## 1. Project overview

This project explores the detection of fake news by fine-tuning a pre-trained XLNet model on the **FakeNewsNet** dataset. The goal is to build an end-to-end NLP pipeline that:

- Cleans and normalises raw news articles.
- Fine-tunes XLNet for binary classification (real vs fake).
- Compares performance against earlier baseline models such as CNN-based and LSTM-based architectures.

This repository contains the original implementation from my Trimester 2 academic project and is kept as an early example of my work with transformer-based NLP.

---

## 2. Key highlights

- **End-to-end pipeline** – from raw text to evaluation: lower-casing, contraction expansion, special-character and stop-word removal, tokenisation and batching.
- **Modern language model** – XLNet fine-tuned for fake-news classification, leveraging its permutation-based language modelling objective and Transformer-XL architecture.
- **Multiple data sources** – experiments run on two subsets of FakeNewsNet:
  - **PolitiFact** – political fact-checking articles.
  - **GossipCop** – entertainment and celebrity news.
- **Baseline comparison** – XLNet compared against:
  - CNN-URG
  - BERT
  - BERT + LSTM
- **Standard evaluation metrics** – accuracy, precision, recall and F1 score to assess model performance.

---

## 3. Dataset

The project uses the **FakeNewsNet** dataset, which provides labelled real and fake news articles from multiple sources.

- **Sources used in this project**
  - **PolitiFact**
  - **GossipCop**
- **Task**
  - Binary classification – predict whether an article is *real* or *fake*.
- **Notes**
  - Articles are in English.
  - Data access and licensing are controlled by the original dataset maintainers – please request and download the data from the official source before running the notebooks.

> This repository does **not** include the raw dataset files for licensing reasons.

---

## 4. Approach

### 4.1 Pre-processing

The pre-processing pipeline (implemented in the notebooks) includes:

- Converting text to lower-case.
- Expanding common contractions (e.g. “don’t” → “do not”).
- Removing special characters and punctuation where appropriate.
- Removing stop words.
- Tokenising text for XLNet.
- Splitting into training, validation and test sets.

### 4.2 Model – XLNet

- Uses a pre-trained **XLNet** model as the base.
- Adds a classification head for binary fake-news detection.
- Fine-tunes the model on PolitiFact and GossipCop articles separately.
- Optimised using standard techniques (learning-rate scheduling, mini-batch training, early stopping based on validation performance).

### 4.3 Baselines

To understand the value of XLNet, the project compares against earlier architectures (implemented outside this repository):

- CNN-based text classifier (CNN-URG).
- BERT fine-tuning.
- BERT + LSTM hybrid.

All models are evaluated on the same splits using accuracy, precision, recall and F1 score.

---

## 5. Experiments and results

### 5.1 XLNet performance

During training, metrics were recorded at the end of each epoch. The table below summarises the strongest epoch for each dataset (by F1 score):

| Dataset     | Best epoch | Accuracy | Precision | Recall | F1 score |
|------------|-----------:|---------:|----------:|-------:|--------:|
| GossipCop  | 2          | 84.4 %   | 78.4 %    | 78.7 % | 78.6 %  |
| PolitiFact | 10         | 88.2 %   | 87.7 %    | 88.3 % | 87.9 %  |

**Observations**

- On **GossipCop**, XLNet stabilises around the mid–high 70s for F1, with accuracy around 84 %. This subset is noisier and more varied (entertainment / celebrity news), so the model’s performance is slightly lower but still strong.
- On **PolitiFact**, XLNet reaches an F1 score of **87.9 %** and accuracy of **88.2 %**, showing that the model handles more structured political fact-checking articles particularly well.
- Precision and recall remain well balanced on both datasets, indicating that the model is not overly biased towards either the real or fake class.

> These results are from single training runs and are intended to illustrate the effectiveness of XLNet on fake-news detection, rather than to represent an exhaustively tuned benchmark.

---

## 6. Repository contents

Current layout:

```text
.
├── README.md                     # Project overview and documentation
├── XLNET_Politifact.ipynb        # XLNet fine-tuning & evaluation on PolitiFact
└── XLNET_GossipCop.ipynb         # XLNet fine-tuning & evaluation on GossipCop

