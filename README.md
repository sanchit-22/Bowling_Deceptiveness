# 🏏 Bowling Deceptiveness Detection using Sequence Models

This repository addresses the task of identifying deceptive bowling deliveries in cricket using sequence-based machine learning models. We evaluate and compare **Logistic Regression**, **Random Forest**, and **LSTM** architectures across various sequence lengths.


## Getting Started

### 🔧 Requirements
- Python 3.7+
- Jupyter / Google Colab
- Libraries: `tensorflow`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`, `pandas`, `numpy`

### Colab Note
All experiments were conducted using **Google Colab**. Ensure that:
- All required `.pkl`, `.h5`, `.csv`, and `.ipynb` files are placed in the **same working directory** (e.g., `/content`).

---

## Training Models

Run the following notebook:

train/smai_proj.ipynb

This will:
- Train Logistic Regression, Random Forest, and LSTM models
- Save models and scalers for each `seq_len ∈ [1, 5]`
- Optionally train enhanced LSTM versions (stacked, weighted loss)

---

## Evaluation

To evaluate model performance:

eval/eval.ipynb

Includes:
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- ROC curves
- Feature importances (RF), coefficient analysis (LR), saliency heatmaps (LSTM)
- Qualitative success/failure analysis
- Model improvement curves

---

## 🔁 Implications to Other Tasks

📄 Run [`train/Implications_Other_Tasks.ipynb`](train/Implications_Other_Tasks.ipynb)

This notebook includes:
- Embedding extraction from trained LSTM (`.h5`)
- Linear probing on learned features using Logistic Regression
- PCA-based dimensionality reduction
- Probing accuracy comparison with original models

Demonstrates that LSTM learns transferable representations, opening doors for:
- Feature reuse in other tasks (zero-shot transfer)
- Low-resource cricket scenarios (linear probing, compression)

---

## Inference Guide

You can test the trained models on **custom, user-defined delivery data** using the notebook:



This script allows interactive prediction by prompting for details about recent deliveries. It supports **sequence lengths from 1 to 5**.

### How to Use

1. **Open `infer/infer.ipynb` in Google Colab**

2. **Provide number of deliveries**  
   You'll be asked:



3. **Manually input delivery features** for each of the `seq_len` entries:
- `landing_x`, `landing_y`
- `ended_x`, `ended_y`
- `ball_speed (in km/h)`
- `ovr` (e.g., `2.3` for 3rd ball of 3rd over)
- `bowler_type` (choose from pre-encoded valid types)

4. **The model automatically:**
- Loads the correct `scaler` and `model` based on your sequence length.
- Transforms the input using the scaler.
- Predicts if the next delivery is **Deceptive** or **Not Deceptive**.
- Displays the model’s confidence score.

### Example Output



### ⚠️ Important

- Ensure all relevant `.pkl` model files (scalers and trained models) are in the same working directory (e.g., `/content` in Colab).
## Key Features

- **Sequential Modeling:** Captures temporal context using up to 5 past deliveries
- **EWMA Labeling:** Automatically detect deceptive deliveries using anomaly detection
- **Ablation Studies:** Effect of sequence length on model performance
- **Interpretability:** Gradient saliency (LSTM), feature importances (RF), weight inspection (LR)
- **Model Transferability:** Experiments with linear probing, dimensionality reduction

---

## Key Files

- Dataset: [`combined.csv`](train/combined.csv)
- Training Script: [`smai_proj.ipynb`](train/smai_proj.ipynb)
- Evaluation Notebook: [`eval.ipynb`](eval/eval.ipynb)
- Inference Notebook: [`infer.ipynb`](infer/infer.ipynb)

---

## Notes

- Run all notebooks in **Google Colab**
- Upload all required files (`.ipynb`, `.pkl`, `.h5`, `combined.csv`) to the same directory (`/content`) before executing

---

## Team 41 - SMAI (Spring 2025)

For academic use only.
