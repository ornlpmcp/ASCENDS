% ASCENDS Lunch & Learn
% Practical Local ML for Scientists
% 30 minutes (Beginner)

# Why ASCENDS

- Local-first workflow (data stays local)
- GUI-first for non-programmers
- Fast baseline + interpretable results

# Session Goal

By the end of this session, you can:

- Select target and input features
- Train a baseline model
- Predict on new data
- Interpret model behavior with SHAP

# Workflow Overview

1. Correlation & Feature Selection
2. Training & Evaluation
3. Prediction
4. SHAP Interpretation

# Demo A: Correlation & Feature Selection

- Upload CSV
- Set target with one click
- Select input features (auto/manual)
- Run Pearson / Spearman / MI / dCor

Key message: correlation is a fast feature-screening step.

# Demo B: Training

- Choose task: regression or classification
- Start with RF baseline
- Run training
- Review outputs:
  - Regression: R2, MAE, RMSE + parity plot
  - Classification: Accuracy, Precision, Recall, F1 + confusion matrix

# Demo C: Prediction + SHAP

- Select a saved model
- Upload new CSV for prediction
- Run SHAP
- Review top driving features

Key message: prediction and interpretation should be used together.

# Common Pitfalls

- Target not set
- Input/target column mismatch
- Train/predict schema mismatch
- Changing seed too often and losing comparability

# Packaging and Release Notes

- Recommended for v0.3.0: pro
- standard is currently experimental/internal
- Linux pro bundle can be larger due to XGBoost/NCCL dependencies

# 30-Minute Agenda

- Intro: 3 min
- Demo A: 7 min
- Demo B: 8 min
- Demo C: 6 min
- Pitfalls: 4 min
- Q&A: 2 min

# Q&A / Next Step

- Start with one small real dataset
- Build one reproducible baseline
- Iterate with domain knowledge + SHAP

Questions?
