# Logistic Regression with 5 Optimizers (From Scratch)

Binary classification using **Logistic Regression** with **degree-2 polynomial features**, trained with five gradient-based optimizers implemented manually in **NumPy**. Each optimizer lives in its own Jupyter Notebook, ready to run on Kaggle or locally.

## Notebooks

| # | Notebook | Content |
|---|----------|---------|
| 0 | [`00_Main_Comparison.ipynb`](00_Main_Comparison.ipynb) | **Main notebook**: all 5 optimizers side by side (metrics table, loss curves, decision boundaries) |
| 1 | [`01_Batch_Gradient_Descent.ipynb`](01_Batch_Gradient_Descent.ipynb) | Batch Gradient Descent |
| 2 | [`02_Stochastic_Gradient_Descent.ipynb`](02_Stochastic_Gradient_Descent.ipynb) | Stochastic Gradient Descent |
| 3 | [`03_MiniBatch_GD_with_Momentum.ipynb`](03_MiniBatch_GD_with_Momentum.ipynb) | Mini-Batch GD + Momentum (β = 0.9) |
| 4 | [`04_RMSprop.ipynb`](04_RMSprop.ipynb) | RMSprop |
| 5 | [`05_Adam.ipynb`](05_Adam.ipynb) | Adam (with bias correction) |

Every notebook is committed **with its outputs** (printed metrics and plots), so the results show directly on GitHub.

## Pipeline

1. **Dataset**: 100 synthetic 2D points generated with NumPy (fixed seed). The label is 1 when the point falls inside a noisy circle, so the classes are not linearly separable. No internet connection is needed.
2. **Preprocessing**: `PolynomialFeatures(degree=2)` → `StandardScaler`, then the bias column is reset to 1.
3. **Model**: Sigmoid hypothesis with Binary Cross-Entropy loss.
4. **Training**: Gradients are computed by hand, `∇J = Xᵀ(σ(Xw) − y) / m`; no ML training libraries are used.
5. **Evaluation**: Accuracy, Precision, Recall, F1 Score and final loss, all computed from the confusion matrix.
6. **Visualization**: The loss curve over epochs and the learned non-linear decision boundary, side by side.

## Update Rules

| Optimizer | Update |
|-----------|--------|
| GD / SGD | `w = w − η·g` |
| Momentum | `v = β·v + η·g`, then `w = w − v` |
| RMSprop | `s = β·s + (1−β)·g²`, then `w = w − η·g / (√s + ε)` |
| Adam | `m = β₁·m + (1−β₁)·g`, `v = β₂·v + (1−β₂)·g²`, `m̂ = m/(1−β₁ᵗ)`, `v̂ = v/(1−β₂ᵗ)`, then `w = w − η·m̂ / (√v̂ + ε)` |

## Results

| Optimizer | Loss | Accuracy | Precision | Recall | F1 |
|-----------|------|----------|-----------|--------|----|
| Batch GD | 0.1225 | 0.98 | 0.9737 | 0.9737 | 0.9737 |
| SGD | 0.0943 | 0.98 | 0.9737 | 0.9737 | 0.9737 |
| Mini-Batch + Momentum | **0.0563** | **0.99** | **1.0000** | 0.9737 | **0.9867** |
| RMSprop | 0.0653 | 0.98 | 0.9737 | 0.9737 | 0.9737 |
| Adam | 0.0814 | 0.98 | 0.9737 | 0.9737 | 0.9737 |

## Requirements

```
numpy
matplotlib
scikit-learn
```

## How to Run

Open any notebook in Kaggle, Colab or Jupyter and choose **Run All**. Each notebook stands on its own and finishes in a few seconds.
