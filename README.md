# Yeast Protein Localization Classifiers

Comparative study of five supervised classification algorithms applied to the UCI *Yeast* dataset, aiming to predict the cellular localization site of yeast proteins from biochemical and sequence-recognition features.

This project was developed for the Machine Learning course (2025.2) at CIn/UFPE under Prof. Francisco de A. T. de Carvalho.

## Problem

The Yeast dataset contains 1,484 protein samples described by 8 continuous predictive attributes. Each protein belongs to one of ten cellular localization classes (e.g., CYT, NUC, MIT, ME3). The task is a multi-class classification with a strongly imbalanced class distribution, which is one of the central challenges addressed throughout the project.

## Dataset

- **Source:** UCI Machine Learning Repository — Yeast Dataset, donated by Paul Horton and created by Kenta Nakai.
- **Reference:** https://archive.ics.uci.edu/dataset/110/yeast
- **Samples:** 1,484
- **Features (8):** `mcg`, `gvh`, `alm`, `mit`, `erl`, `pox`, `vac`, `nuc`
- **Target:** `site` — cellular localization class
- A local copy of the dataset is included under `data/yeast-datasets.zip`.

## Methodology

The experimental pipeline follows the project specification:

1. **Exploratory analysis** — univariate distributions, bivariate correlations, class imbalance assessment.
2. **Preprocessing** — label encoding of the target variable and standardization of the predictors via `StandardScaler`.
3. **Stratified split** — 70% training/validation and 30% test, preserving class proportions.
4. **Model training** — five classifiers are trained:
   - Decision Tree
   - Gaussian Naive Bayes
   - Logistic Regression
   - k-Nearest Neighbors (Euclidean distance, `k` chosen via cross-validation)
   - Random Forest
5. **Hyperparameter tuning** — `GridSearchCV` with stratified k-fold cross-validation. The test set is never used during tuning.
6. **Evaluation** — precision, recall, F1-score, and per-class classification reports.
7. **Learning curves** — each best-tuned classifier is evaluated across training/test proportions from (5%, 95%) to (95%, 5%) in 5% steps, using stratified sampling.

## Summary of Results

| Classifier          | Test Accuracy |
|---------------------|---------------|
| Decision Tree       | 54%           |
| Naive Bayes         | 12%           |
| Logistic Regression | 58%           |
| k-Nearest Neighbors | 56%           |
| Random Forest       | best overall  |

Random Forest achieved the best and most stable performance. Naive Bayes performed poorly, which is consistent with the violation of the conditional independence assumption for this dataset. Class imbalance affected every model and is the dominant factor limiting overall accuracy.

## Repository Layout

```
.
├── README.md
├── .gitignore
├── notebooks/
│   └── yeast_classification.ipynb   # Full experimental pipeline
├── docs/
│   ├── project-brief.pdf            # Original course assignment
│   └── presentation.pdf             # Final presentation slides
└── data/
    └── yeast-datasets.zip           # Local copy of the Yeast dataset
```

## How to Run

The notebook runs on Python 3.10+ with the scientific Python stack:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn plotly jupyter
jupyter notebook notebooks/yeast_classification.ipynb
```

The notebook expects the dataset to be available locally. If running outside Google Colab, extract `data/yeast-datasets.zip` and adjust the data-loading cell accordingly.

## Authors

- Andreywid Yago Lima de Souza
- Artur Vinicius Pereira Fernandes
- Heitor Higino Holanda Cordeiro
- Arthur Thierre Dos Santos Silva

Centro de Informatica (CIn), Universidade Federal de Pernambuco (UFPE).

## References

- Horton, P. and Nakai, K. *A Probabilistic Classification System for Predicting the Cellular Localization Sites of Proteins.* UCI Machine Learning Repository.
- Pedregosa et al. *Scikit-learn: Machine Learning in Python*, JMLR 12, 2011.
