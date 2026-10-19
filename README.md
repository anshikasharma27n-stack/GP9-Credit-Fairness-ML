# Fairness, Calibration and Explainability in ML Credit Default Prediction (Group 9)

**Group members:** Syed Yunus Rizvi, Rishika Chaurasia, Anshika Sharma, Gaurav Krishna (PGDM 2025-27, Jaipuria Institute of Management, Noida)

Reproducible notebook for the ML group project: Logistic Regression, Random Forest and XGBoost compared on
discrimination (ROC-AUC, PR-AUC, F1), calibration (Brier, ECE), subgroup error rates and fairness gaps across
recorded demographic groups (sex, age band, marital status, education), with SHAP explanations,
threshold-based fairness mitigation, multi-seed robustness and a cost analysis.

## Dataset
- Default of Credit Card Clients (Taiwan, 2005), 30,000 clients, 23 features, target = default next month.
- Source: Yeh, I. C., & Lien, C. H. (2009). UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H
- Copy used: Kaggle `uciml/default-of-credit-card-clients-dataset`, file `UCI_Credit_Card.csv`
  (https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset).
- The exact file used is already included in this repository at `data/UCI_Credit_Card.csv`; no download is needed.

## How to reproduce
Requires **Python 3.13** (the version used for the reported results; library versions are pinned in `requirements.txt`).
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute GP9_Credit_Fairness_ML.ipynb --output GP9_Credit_Fairness_ML.ipynb
```
Runtime: about 6 minutes on a 2-core machine. All tables are written to `results/`, all figures to `figures/`.

## Key design decisions
| Item | Choice |
|---|---|
| Random seed | 42 (robustness check: seeds 0-4) |
| Split | Stratified 60% train / 20% validation / 20% test, done before any pre-processing |
| Cleaning | EDUCATION codes 0, 5, 6 -> 4 (others); MARRIAGE 0 -> 3 (others); 35 exact duplicate records removed |
| Missing values / outliers | None missing; engineered ratios are clipped to fixed ranges |
| Engineered features | AVG_UTIL, PAY_TO_BILL, N_DELAYS, MAX_DELAY, UTIL_TREND |
| Pre-processing | sklearn Pipeline + ColumnTransformer, fitted on training folds only (scaling for LR only; one-hot for SEX, EDUCATION, MARRIAGE) |
| Models | Dummy baseline, Logistic Regression, Random Forest, XGBoost |
| Tuning | RandomizedSearchCV, 5-fold stratified CV, scoring = average precision (PR-AUC) |
| Class imbalance | No re-weighting in main models; decision threshold chosen on validation set (max F1). Class-weighted XGBoost tested as an ablation |
| Calibration | Brier, ECE (10 bins), reliability curves, isotonic re-calibration on validation set |
| Fairness | Approval rate, FPR, FNR, AUC, ECE per group; SPD, DIR, EOD (FPR gap), equalised odds; bootstrap CIs; unawareness ablation; global and group-specific thresholds |
| Explainability | SHAP TreeExplainer on XGBoost (global and local) |

## Repository structure
```
data/UCI_Credit_Card.csv          dataset (from Kaggle)
GP9_Credit_Fairness_ML.ipynb      executed notebook
GP9_Credit_Fairness_ML.html       HTML export
build_nb.py                       script that generates the notebook
results/                          all result tables (CSV) + summary.json
figures/                          all figures (PNG)
requirements.txt                  library versions
```
