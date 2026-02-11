# ML-Classification-Playground
Machine learning classification algorithms applied to real-world challenges. Dedicated to the implementation, evaluation, and comparison of supervised machine learning classification algorithms using PyTorch.

## Objective

This project aims to:

- Implement classification algorithms
- Compare models using robust evaluation metrics
- Explore feature engineering techniques
- Apply models to real-world datasets
- Build reproducible ML pipelines

---

## Algorithms Covered

- Softmax 1D
- SoftMax 2D

---

## Evaluation Metrics

- Accuracy
- Precision / Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Cross-validation
- Stratified K-Fold

---

## Project Structure

    classification-lab/
    │
    ├── data/
    │   ├── raw/
    │   └── processed/
    │
    ├── notebooks/
    │
    ├── src/
    │   ├── softmax/
    │   │   ├── softmax_1d/
    │   │   │   ├── model.py
    │   │   │   ├── loss.py
    │   │   │   ├── gradients.py
    │   │   │   └── train.py
    │   │   │
    │   │   ├── softmax_2d/
    │   │   │   ├── model.py
    │   │   │   ├── loss.py
    │   │   │   ├── gradients.py
    │   │   │   └── train.py
    │   │   │
    │   │   └── utils.py
    │   │
    │   ├── evaluation/
    │   │   ├── metrics.py
    │   │   ├── confusion_matrix.py
    │   │   └── cross_validation.py
    │   │
    │   └── preprocessing/
    │       ├── scaling.py
    │       ├── encoding.py
    │       └── feature_engineering.py
    │
    ├── tests/
    │   ├── test_softmax_1d.py
    │   ├── test_softmax_2d.py
    │   └── test_metrics.py
    │
    ├── requirements.txt
    ├── README.md
    └── LICENSE
