# Explainable Housing Price Prediction Using Random Forest and Neural Networks

This repository contains the source files and empirical analysis for my master's thesis in **Advanced Analytics - Big Data** at the **Warsaw School of Economics**.

**Author:** Mengzhen Shang  
**Supervisor:** Dr Grzegorz Koloch  
**Year:** 2026

[Read the full thesis](Mengzhen_Shang_Master_Thesis.pdf)

## Project Overview

Housing valuation models must do more than produce accurate estimates when their outputs support mortgage lending, investment analysis, pricing recommendations, or automated valuation systems. They should also provide explanations that are understandable, auditable, and consistent with real-estate logic.

This project evaluates explainable machine learning for housing price prediction using the Ames Housing dataset. It focuses on the trade-off between predictive accuracy and interpretability by comparing Random Forest with a Multi-Layer Perceptron neural network. Linear Regression, Ridge Regression, and XGBoost are included as benchmarks.

## Research Questions

The study asks:

1. How accurately can different machine-learning models predict housing prices after preprocessing and feature engineering?
2. How does a tuned Random Forest compare with a tuned neural network in predictive performance and generalization?
3. How do Linear Regression, Ridge Regression, and XGBoost perform relative to the two main models?
4. Which housing characteristics are most important according to global and local SHAP explanations?
5. Does Random Forest provide stable and economically meaningful explanations for individual predictions?
6. How robust is Random Forest when the engineered feature `TotalSF` is removed?
7. What are the practical implications of explainable housing valuation for banks, real-estate platforms, buyers, sellers, and analysts?

## Data

The analysis uses the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) competition data, which is based on the Ames Housing dataset compiled by Dean De Cock.

The original training data contain:

- 1,460 residential property observations;
- 80 explanatory variables covering structure, quality, location, and sale conditions;
- `SalePrice` as the prediction target.

The dataset is not included in this repository. Download `train.csv` from Kaggle, rename it to `house_price.csv`, and place it in the repository root before running the analysis.

## Data Cleaning and Feature Engineering

The preprocessing workflow performs the following steps:

1. Removes the non-informative `Id` variable.
2. Removes two extreme observations where `GrLivArea > 4000` and `SalePrice < 300000`.
3. Applies `log1p` to `SalePrice` to reduce right skewness and stabilize variance.
4. Encodes structural absence as `"None"` for categorical variables such as basement, garage, alley, pool, and fireplace features.
5. Uses median imputation for selected skewed numerical variables, including `LotFrontage`, `GarageYrBlt`, and `MasVnrArea`.
6. Uses mode imputation for categorical variables with a small number of missing values.
7. Replaces missing structural quantities with zero where absence has a clear physical meaning.
8. Engineers three features:
   - `HouseAge = YrSold - YearBuilt`
   - `RemodAge = YrSold - YearRemodAdd`
   - `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
9. Applies one-hot encoding to categorical variables.

After cleaning and encoding, the modelling dataset contains **1,458 observations and 261 predictors**.

## Models Compared

Five regression models are evaluated:

- Linear Regression
- Ridge Regression
- XGBoost
- Tuned Random Forest
- Tuned Multi-Layer Perceptron neural network

Random Forest and the neural network are the two principal models. Linear Regression and Ridge Regression provide simpler statistical benchmarks, while XGBoost provides a strong tree-boosting benchmark for structured tabular data.

## Training and Evaluation

- The processed data are split into **80% training data and 20% hold-out test data**.
- `random_state=42` is used for reproducibility.
- Random Forest and the neural network are tuned with `GridSearchCV` using **5-fold cross-validation** on the training set only.
- Standardization for the neural network is performed inside a scikit-learn pipeline, preventing information leakage across validation folds.
- The hold-out test set is used only for final model comparison.
- Models are evaluated using RMSE, MAE, and R-squared on the log-transformed target.

## Model Performance

| Model | RMSE | MAE | R-squared |
|---|---:|---:|---:|
| XGBoost | **0.1238** | **0.0852** | **0.9091** |
| Ridge Regression | 0.1340 | 0.0917 | 0.8934 |
| Linear Regression | 0.1371 | 0.0934 | 0.8884 |
| Tuned Random Forest | 0.1454 | 0.0975 | 0.8747 |
| Tuned Neural Network | 0.9996 | 0.7119 | -4.9277 |

All metrics are calculated on the same hold-out test set using log-transformed `SalePrice`.

### Why XGBoost Performed Best

XGBoost achieved the lowest prediction errors and the highest R-squared. Its sequential boosting process allows each new tree to focus on errors left by previous trees. This is particularly effective for structured housing data, where price depends on nonlinear relationships and interactions among variables such as size, quality, neighborhood, renovation status, and garage capacity.

Random Forest trains trees independently and averages their predictions. This bagging mechanism provides stable results but may smooth over some of the finer pricing patterns that boosting can capture. XGBoost was therefore the strongest model when predictive accuracy was the primary objective.

## Why the Thesis Focuses on Random Forest and the Neural Network

The main research objective is broader than finding the model with the lowest RMSE. Random Forest and the neural network represent two different learning approaches:

- **Random Forest** is well suited to medium-sized tabular data, captures nonlinear relationships, requires limited scaling, and can be interpreted effectively with SHAP.
- **The neural network** represents a more flexible and higher-complexity approach, allowing the study to test whether additional model complexity improves performance and explanation quality.

XGBoost is included as a strong predictive benchmark. Random Forest remains central because it offers a useful balance between accuracy, interpretability, and robustness. The neural network's weak performance is also an informative result: greater model complexity did not improve generalization on this medium-sized, high-dimensional, one-hot-encoded dataset.

## SHAP Findings

SHAP is used to evaluate both global feature importance and individual predictions.

### Random Forest

The Random Forest explanations are concentrated around economically meaningful housing attributes. The five most important features are:

| Feature | Mean absolute SHAP value |
|---|---:|
| `TotalSF` | 0.1319 |
| `OverallQual` | 0.1018 |
| `GarageCars` | 0.0146 |
| `GarageArea` | 0.0139 |
| `GrLivArea` | 0.0129 |

Higher total floor area and higher overall quality generally increase predicted log price, while smaller, older, or lower-quality properties receive negative contributions. Other influential variables include `HouseAge`, `BsmtFinSF1`, `RemodAge`, `YearBuilt`, and `LotArea`.

Local explanations for the best, typical, and worst prediction cases show that the model adjusts its reasoning to each property's characteristics. Large floor area and strong quality increase valuation, while limited size, lower quality, older age, lack of central air conditioning, and weaker garage features reduce it.

### Neural Network

The neural network produces a more fragmented importance profile and relies more heavily on sparse categorical indicators, including masonry veneer type, detached garage type, neighborhood, exterior materials, and pool quality. Because the neural network generalizes poorly, these explanations should be treated primarily as a diagnostic comparison rather than reliable evidence about housing value.

### Robustness Check

Removing `TotalSF`, the most important Random Forest feature, produces almost no change in hold-out performance:

| Random Forest specification | RMSE | MAE | R-squared |
|---|---:|---:|---:|
| Original model | 0.145357 | 0.097450 | 0.874665 |
| Without `TotalSF` | 0.145416 | 0.097459 | 0.874562 |

This indicates that Random Forest does not depend exclusively on one engineered variable. It can recover similar information from related features such as above-ground living area, basement area, floor-specific area, and garage variables.

SHAP values describe the model's predictive associations; they should not be interpreted as causal effects.

## Limitations

- Ames represents one local housing market, so the findings may not generalize to other cities or countries.
- The dataset is small relative to its 261 encoded predictors, particularly for neural-network modelling.
- The data contain limited time and micro-location information and omit macroeconomic conditions.
- Property photographs, floor plans, listing text, interior condition, sunlight, noise, views, and undocumented renovations are unavailable.
- XGBoost is used mainly as a benchmark and does not receive the same detailed SHAP analysis as Random Forest.
- The neural-network analysis covers a standard MLP rather than more advanced architectures designed for tabular data.
- Hyperparameter search is constrained by time and computational cost.
- Local SHAP analysis covers only three representative cases, and the robustness test removes only `TotalSF`.
- Fairness, user comprehension, production monitoring, legal compliance, and real-world deployment are outside the empirical scope of the thesis.

## Repository Structure

```text
.
├── figure/                         # Figures used in the thesis
├── housing_price_analysis.py       # Complete empirical analysis
├── main.tex                        # LaTeX thesis source
├── Mengzhen_Shang_Master_Thesis.pdf
├── README.md
├── requirements.txt
└── sgh_full.png                    # University logo used on the title page
```

## Reproduction

### 1. Clone the repository

```bash
git clone https://github.com/MaggieShang/masters-thesis-housing-price-prediction.git
cd masters-thesis-housing-price-prediction
```

### 2. Create and activate a virtual environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the data

Download `train.csv` from the [Kaggle competition data page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data), rename it to `house_price.csv`, and place it in the repository root:

```text
masters-thesis-housing-price-prediction/
├── house_price.csv
├── housing_price_analysis.py
└── ...
```

### 5. Run the analysis

```bash
python housing_price_analysis.py
```

The script performs preprocessing, model tuning, cross-validation, hold-out evaluation, SHAP analysis, and the `TotalSF` robustness check. It also creates result tables and displays the figures used in the thesis. Grid search and Kernel SHAP may take several minutes depending on the computer.

## Main Libraries

- pandas
- NumPy
- Matplotlib
- seaborn
- scikit-learn
- XGBoost
- SHAP

## Citation

If you refer to this work, please cite:

```text
Shang, M. (2026). Explainable Housing Price Prediction Using Random Forest
and Neural Networks: Evidence from the Ames Housing Dataset. Master's thesis,
Warsaw School of Economics.
```

## Copyright

Copyright (c) 2026 Mengzhen Shang. All rights reserved.
