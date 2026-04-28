# 🕵️ Fake News Detection with DistilBERT + Stylistic Feature Fusion

A deep learning pipeline for fake news classification that combines **transformer-based semantic embeddings** with **handcrafted stylistic features** — capturing not just *what* an article says, but *how* it is written.

---

## 📌 Research Question

> **Does combining semantic embeddings with stylistic features improve fake news detection compared to using semantic features alone?**

**Answer: Yes.** Our fusion model achieves **99.17% accuracy** and a **99.08% F1 score**, outperforming a standard TF-IDF baseline (~95%) by over 4 percentage points across all metrics.

---

## 🗂️ Project Structure

```
├── Data_Mining_final.ipynb   # Full pipeline: EDA → Feature Engineering → Model → Evaluation
├── WELFake_Dataset.csv       # Dataset (see Dataset section below)
└── README.md
```

---

## 📦 Dataset

**WELFake Dataset**
- A widely used benchmark for fake news detection containing news articles with title, body text, and binary labels.
- **Label convention used in this project:** `0 = Real`, `1 = Fake`
- Input features used: article title + body text (combined), no images or URLs.

---

## 🔧 Pipeline Overview

### 1. Exploratory Data Analysis
- Class distribution check
- Article length analysis (title and body word count) by class
- Top unigrams and bigrams per class (TF-IDF based)

### 2. Feature Engineering

#### Semantic Features
- Text is tokenized and encoded using **DistilBERT** (`distilbert-base-uncased`), producing a 768-dimensional `[CLS]` embedding per article.

#### Stylistic Features (7 handcrafted signals)
| Feature | Description |
|---|---|
| Sentiment Polarity | Overall emotional tone of the text |
| Subjectivity | Degree of opinion vs. factual language |
| Readability | Flesch reading ease score |
| Exclamation Count | Number of `!` marks |
| Question Count | Number of `?` marks |
| Caps Ratio | Proportion of ALL CAPS words |
| Lexical Diversity | Type-token ratio (vocabulary variety) |

Fake news was found to have **higher subjectivity, simpler readability, more punctuation, greater ALL CAPS usage, and lower lexical diversity** than real news — consistent with content designed to provoke rather than inform.

### 3. Model Architecture: Late Fusion

```
Text Input
    │
    ▼
DistilBERT Encoder
    │
[CLS] Embedding (768-dim)
    │                         Stylistic Features (7-dim)
    │                                │
    │                         Linear(7 → 32) + ReLU + Dropout
    │                                │
    └──────────── Concat (800-dim) ──┘
                        │
               Linear(800 → 128) + ReLU + Dropout
                        │
               Linear(128 → 2)  →  Fake / Real
```

Each modality has a **dedicated encoding pathway** before fusion, ensuring stylistic signals are not drowned out by the high-dimensional transformer output.

### 4. Training

| Setting | Value |
|---|---|
| Base Model | `distilbert-base-uncased` |
| Optimizer | AdamW |
| Learning Rate | 2e-5 |
| Batch Size | 16 |
| Epochs | 3 |
| Loss | CrossEntropyLoss |
| Train / Val Split | 80% / 20% (stratified) |
| Max Token Length | 128 |

Training loss: `0.0572 → 0.0169 → 0.0075` — steady convergence with no instability.

### 5. Uncertainty Estimation: MC Dropout
During inference, dropout is kept active and the model runs **T = 10** forward passes per sample. The mean probability is used as the final prediction and the variance serves as an **uncertainty score** — flagging borderline cases that may warrant human review.

---

## 📊 Results

| Metric | Baseline (TF-IDF + LR) | Our Model (DistilBERT + Fusion) |
|---|---|---|
| Accuracy | ~95.00% | **99.17%** |
| Precision | ~95.00% | **98.99%** |
| Recall | ~95.00% | **99.17%** |
| F1 Score | ~95.00% | **99.08%** |

---

## ⚠️ Limitations

- **Text-only:** The model does not process images or URLs, both of which are important vectors for misinformation.
- **Static dataset:** Performance may degrade on newer fake news styles without retraining.
- **No URL signals:** Domain reputation and URL structure are strong credibility indicators that are currently unused.

---

## 🔭 Future Work

The natural next step is a **multimodal fake news detector** that extends this late fusion design with:

- **Vision encoder (CLIP / ViT)** — detect image manipulation and image-text inconsistencies
- **URL feature encoder** — extract domain reputation, TLD type, and structural URL signals
- **Cross-modal consistency modeling** — explicitly penalize mismatches between visual and textual content

---

## 🛠️ Requirements

```bash
pip install -r requirements.txt
```

> Recommended: Run on **Google Colab** with GPU runtime for faster DistilBERT fine-tuning.

---

## 🚀 How to Run

1. Clone the repository and place `WELFake_Dataset.csv` in the root directory.
2. Open `main_notebook.ipynb` in Jupyter or Google Colab.
3. Run all cells top to bottom — EDA, feature engineering, training, and evaluation are all self-contained.

---

## 📚 Key References

- Devlin et al. (2019) — *BERT: Pre-training of Deep Bidirectional Transformers*
- Sanh et al. (2019) — *DistilBERT, a distilled version of BERT*
- Gal & Ghahramani (2016) — *Dropout as a Bayesian Approximation (MC Dropout)*
- WELFake Dataset — Verma et al. (2021)
