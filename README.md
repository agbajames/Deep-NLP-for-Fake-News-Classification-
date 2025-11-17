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

> This repository does **not** include the raw dataset.

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
├── README.md
└── notebooks/
    ├── XLNET_Politifact.ipynb
    └── XLNET_GossipCop.ipynb
```

Each notebook is self-contained and walks through:

1. Loading and cleaning the dataset.
2. Tokenising text for XLNet.
3. Fine-tuning the model.
4. Evaluating performance and printing metrics.

---

## 7. Getting started

### 7.1 Prerequisites

- Python 3.x  
- `pip` or `conda`  
- Access to the FakeNewsNet dataset (PolitiFact and/or GossipCop)

### 7.2 Suggested setup

Create a virtual environment and install the required packages:

```bash
git clone https://github.com/agbajames/fake-news-detection-xlnet.git
cd fake-news-detection-xlnet

python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scriptsctivate

# Install core dependencies (add or adjust based on your environment)
pip install torch torchvision torchaudio
pip install transformers
pip install scikit-learn pandas numpy jupyter
```

Download the relevant FakeNewsNet subsets and update the dataset paths in the notebooks where indicated.

Then launch Jupyter:

```bash
jupyter notebook
```

Open either `XLNET_Politifact.ipynb` or `XLNET_GossipCop.ipynb` and run the cells top to bottom.

---

## 8. Limitations and future directions

This project represents an early academic exploration of transformer-based fake-news detection and has a few limitations:

- Experiments are run in notebooks only – no production-ready scripts or API.
- No formal experiment tracking (e.g. MLflow, Weights & Biases).
- Evaluation is focused on standard metrics; no calibration analysis or robustness checks.
- More recent models (e.g. RoBERTa, DeBERTa, modern LLMs) are not included.

If I were to revisit this work today, likely next steps would include:

- Re-implementing the pipeline as reusable Python modules with a CLI.
- Adding MLflow/W&B tracking for experiments.
- Comparing XLNet with more recent transformer architectures.
- Exposing the best model behind a simple REST API or demo app.

---

## 9. What I learnt

Even though this is an early project, it helped me build foundations that I now use on larger NLP and LLM systems:

- Designing a full NLP pipeline for noisy, real-world news text.
- Fine-tuning pre-trained transformer models for downstream tasks.
- Comparing different model families on the same task using consistent metrics.
- Understanding the impact of pre-processing decisions on downstream performance.

---

## 10. Project status

> **Status – archived (original academic version)**  

This repository is kept as a record of my Trimester 2 project on fake-news detection using XLNet. I now work with larger-scale NLP and LLM systems, but this project remains a useful snapshot of my early transformer-based work.
