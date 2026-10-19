import nbformat as nbf

cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
def code(s): cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Fairness, Calibration and Explainability in Machine-Learning Credit Default Prediction
### Group 9: Syed Yunus Rizvi, Rishika Chaurasia, Anshika Sharma, Gaurav Krishna (Jaipuria Institute of Management, PGDM 2025-27)

**Research problem.** Banks use machine-learning (ML) models to predict whether a credit-card client will default next month. A model can be accurate overall and still (a) give probabilities that do not match real default rates (poor calibration) and (b) make more mistakes for some demographic groups than others (unfairness).

**Objective.** Compare Logistic Regression, Random Forest and gradient boosting (XGBoost) on predictive performance (ROC-AUC, PR-AUC, F1), calibration (Brier score, ECE, reliability curves) and subgroup error rates across the demographic groups recorded in the data (sex, age band, marital status, education). Explain the models with SHAP and test whether threshold choices can reduce group gaps.

**Research questions**
- **RQ1.** Which model gives the best discrimination (ROC-AUC, PR-AUC, F1) compared with a naive baseline?
- **RQ2.** Are the models' probabilities well calibrated, overall and within each demographic group, and does imbalance handling affect calibration?
- **RQ3.** Do error rates (wrongful rejection of good borrowers, missed defaulters) and approval rates differ across sex, age, marital status and education?
- **RQ4.** Does removing demographic variables ("fairness through unawareness") or adjusting decision thresholds reduce these gaps, and at what cost to accuracy?
- **RQ5.** Which features drive predictions globally and for individual borrowers (SHAP), and do demographic variables matter?

**Fairness rule used in this study.** We only use demographic variables that are recorded in the dataset (SEX, AGE, MARRIAGE, EDUCATION). We do not infer or guess any protected trait that is not present.

**Notebook structure:** 0 Setup → 1 Data & provenance → 2 Cleaning → 3 EDA → 4 Feature engineering → 5 Split → 6 Pre-processing pipelines → 7 Model development & tuning → 8 Test evaluation → 9 Calibration → 10 Subgroup fairness → 11 Mitigation (unawareness & thresholds) → 12 SHAP → 13 Robustness across seeds → 14 Business cost analysis → 15 Summary
""")

md("## 0. Setup and reproducibility\nAll random steps use one seed (`SEED = 42`). Library versions are printed so results can be reproduced.")
code(r"""
import warnings, sys, os, json, time
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn, xgboost, shap

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn import metrics
from xgboost import XGBClassifier
from scipy.stats import loguniform, randint, uniform

SEED = 42
np.random.seed(SEED)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight"})

print("Python      :", sys.version.split()[0])
for lib in [np, pd, matplotlib, sns, sklearn, xgboost, shap]:
    print(f"{lib.__name__:<12}:", lib.__version__)
""")

md(r"""
## 1. Data and provenance
**Dataset:** *Default of Credit Card Clients* (Taiwan, April to September 2005). 30,000 credit-card clients, 23 explanatory variables and a binary target (default next month).
- Original source: Yeh, I. C., & Lien, C. H. (2009). *Expert Systems with Applications, 36*(2), 2473-2480. Donated to the UCI Machine Learning Repository: https://doi.org/10.24432/C55S3H
- Copy used here: Kaggle, `uciml/default-of-credit-card-clients-dataset`, file `UCI_Credit_Card.csv` (https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

| Variable | Meaning |
|---|---|
| LIMIT_BAL | Credit limit (NT dollars), individual and family |
| SEX | 1 = male, 2 = female |
| EDUCATION | 1 = graduate school, 2 = university, 3 = high school, 4 = others (0, 5, 6 undocumented) |
| MARRIAGE | 1 = married, 2 = single, 3 = others (0 undocumented) |
| AGE | Age in years |
| PAY_0, PAY_2 ... PAY_6 | Repayment status Sept ... April (-2 no consumption, -1 paid in full, 0 revolving credit, 1-9 months of delay) |
| BILL_AMT1-6 | Bill statement amount Sept ... April |
| PAY_AMT1-6 | Amount paid Sept ... April |
| default.payment.next.month | Target: 1 = default in October 2005 |
""")
code(r"""
DATA_PATH = "data/UCI_Credit_Card.csv"
raw = pd.read_csv(DATA_PATH)
print("Shape:", raw.shape)
print("Missing values:", int(raw.isna().sum().sum()))
raw.head()
""")

md("## 2. Data cleaning\nWe (1) rename columns for clarity, (2) recode undocumented category codes into the documented 'others' category, and (3) remove exact duplicate client records so that the same record cannot appear in both training and test sets (a form of leakage).")
code(r"""
df = raw.rename(columns={"PAY_0": "PAY_1", "default.payment.next.month": "DEFAULT"}).copy()

print("EDUCATION codes before:", df["EDUCATION"].value_counts().sort_index().to_dict())
print("MARRIAGE codes before :", df["MARRIAGE"].value_counts().sort_index().to_dict())
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})   # undocumented -> 'others'
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})                 # undocumented -> 'others'

feature_cols_raw = [c for c in df.columns if c not in ["ID", "DEFAULT"]]
n_dup = df.duplicated(subset=feature_cols_raw + ["DEFAULT"]).sum()
df = df.drop_duplicates(subset=feature_cols_raw + ["DEFAULT"]).reset_index(drop=True)
print(f"Exact duplicate records removed: {n_dup}")
print("Final shape:", df.shape)
""")

code(r"""
# Readable labels for the demographic (sensitive) attributes, used only for fairness analysis
df["SEX_LBL"] = df["SEX"].map({1: "Male", 2: "Female"})
df["EDU_LBL"] = df["EDUCATION"].map({1: "Graduate school", 2: "University", 3: "High school", 4: "Others"})
df["MAR_LBL"] = df["MARRIAGE"].map({1: "Married", 2: "Single", 3: "Others"})
df["AGE_BAND"] = pd.cut(df["AGE"], bins=[20, 29, 39, 49, 100], labels=["21-29", "30-39", "40-49", "50+"])
SENSITIVE = {"Sex": "SEX_LBL", "Age band": "AGE_BAND", "Marital status": "MAR_LBL", "Education": "EDU_LBL"}
""")

md("## 3. Exploratory data analysis (EDA)")
code(r"""
rate = df["DEFAULT"].mean()
print(f"Default rate: {rate:.2%}  (non-default {1-rate:.2%})  -> imbalanced, about 3.5 : 1")

eda_rows = []
for name, col in SENSITIVE.items():
    g = df.groupby(col, observed=True)["DEFAULT"].agg(["count", "mean"]).reset_index()
    g.columns = ["Group", "n", "Default rate"]
    g.insert(0, "Attribute", name)
    g["Share of sample"] = g["n"] / len(df)
    eda_rows.append(g)
eda_groups = pd.concat(eda_rows, ignore_index=True)
eda_groups.to_csv("results/table_eda_default_rate_by_group.csv", index=False)
eda_groups
""")
code(r"""
fig, axes = plt.subplots(1, 4, figsize=(17, 3.8), sharey=True)
for ax, (name, col) in zip(axes, SENSITIVE.items()):
    g = df.groupby(col, observed=True)["DEFAULT"].mean()
    ax.bar(g.index.astype(str), g.values, color="#4C72B0")
    ax.axhline(rate, color="crimson", ls="--", lw=1, label="Overall")
    ax.set_title(f"Default rate by {name.lower()}")
    ax.tick_params(axis="x", rotation=30)
axes[0].set_ylabel("Observed default rate"); axes[0].legend()
plt.tight_layout(); plt.savefig("figures/fig1_default_rate_by_group.png"); plt.show()
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
df["DEFAULT"].value_counts().sort_index().plot.bar(ax=axes[0], color=["#55A868", "#C44E52"])
axes[0].set_xticklabels(["No default (0)", "Default (1)"], rotation=0); axes[0].set_title("Class distribution")
sns.boxplot(data=df, x="DEFAULT", y="LIMIT_BAL", ax=axes[1], showfliers=False)
axes[1].set_title("Credit limit by default status")
pd.crosstab(df["PAY_1"], df["DEFAULT"], normalize="index")[1].plot.bar(ax=axes[2], color="#C44E52")
axes[2].set_title("Default rate by latest repayment status (PAY_1)"); axes[2].set_ylabel("Default rate")
plt.tight_layout(); plt.savefig("figures/fig2_eda_target_limit_pay.png"); plt.show()
""")
code(r"""
num_for_corr = ["LIMIT_BAL", "AGE"] + [f"PAY_{i}" for i in range(1, 7)] + ["BILL_AMT1", "PAY_AMT1", "DEFAULT"]
plt.figure(figsize=(9, 7))
sns.heatmap(df[num_for_corr].corr(), cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 7})
plt.title("Correlation matrix (selected variables)")
plt.savefig("figures/fig3_correlation.png"); plt.show()
""")

md("## 4. Feature engineering\nFive row-level features are created from each client's own history (no information from other rows is used, so no leakage).\n- `AVG_UTIL`: average bill amount divided by credit limit (how much of the limit is used)\n- `PAY_TO_BILL`: total paid divided by total billed over six months (capped at 0 to 5)\n- `N_DELAYS`: number of months with a payment delay (PAY > 0)\n- `MAX_DELAY`: worst delay in the six months\n- `UTIL_TREND`: latest bill minus oldest bill, divided by limit (is debt rising?)")
code(r"""
bill = [f"BILL_AMT{i}" for i in range(1, 7)]
pay_amt = [f"PAY_AMT{i}" for i in range(1, 7)]
pay_st = [f"PAY_{i}" for i in range(1, 7)]

df["AVG_UTIL"] = (df[bill].mean(axis=1) / df["LIMIT_BAL"]).clip(-1, 5)
df["PAY_TO_BILL"] = (df[pay_amt].sum(axis=1) / df[bill].sum(axis=1).clip(lower=1)).clip(0, 5)
df["N_DELAYS"] = (df[pay_st] > 0).sum(axis=1)
df["MAX_DELAY"] = df[pay_st].max(axis=1).clip(lower=0)
df["UTIL_TREND"] = ((df["BILL_AMT1"] - df["BILL_AMT6"]) / df["LIMIT_BAL"]).clip(-5, 5)

ENG = ["AVG_UTIL", "PAY_TO_BILL", "N_DELAYS", "MAX_DELAY", "UTIL_TREND"]
df[ENG + ["DEFAULT"]].groupby("DEFAULT").mean()
""")

md("## 5. Train / validation / test split\nStratified **60% train, 20% validation, 20% test**, seed 42. The split happens **before** any scaling, encoding or tuning.\n- **Train**: fit models and pre-processing (with 5-fold cross-validation for tuning)\n- **Validation**: choose decision thresholds and fit calibration\n- **Test**: touched only once for the final, honest evaluation")
code(r"""
NUM_FEATS = ["LIMIT_BAL", "AGE"] + pay_st + bill + pay_amt + ENG
CAT_FEATS = ["SEX", "EDUCATION", "MARRIAGE"]
ALL_FEATS = NUM_FEATS + CAT_FEATS
TARGET = "DEFAULT"

def make_split(data, seed):
    tr, te = train_test_split(data, test_size=0.20, stratify=data[TARGET], random_state=seed)
    tr, va = train_test_split(tr, test_size=0.25, stratify=tr[TARGET], random_state=seed)   # 0.25 x 0.8 = 0.2
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)

train, val, test = make_split(df, SEED)
for n, d in [("Train", train), ("Validation", val), ("Test", test)]:
    print(f"{n:<11} n = {len(d):>6,}   default rate = {d[TARGET].mean():.4f}")
X_tr, y_tr = train[ALL_FEATS], train[TARGET]
X_va, y_va = val[ALL_FEATS], val[TARGET]
X_te, y_te = test[ALL_FEATS], test[TARGET]
""")

md("## 6. Pre-processing pipelines\nPre-processing lives **inside** sklearn `Pipeline`s, so it is learned only from the training folds (no leakage).\n- Logistic Regression: numeric features standardised; SEX, EDUCATION, MARRIAGE one-hot encoded.\n- Tree models (Random Forest, XGBoost): numeric features passed through unchanged (trees do not need scaling); categories one-hot encoded.\n\n**Class imbalance.** The main models are trained **without** re-weighting, and imbalance is handled by **choosing the decision threshold on the validation set**. This keeps predicted probabilities interpretable as real default probabilities. Section 9 tests what re-weighting does to calibration.")
code(r"""
def make_pre(scale, num=NUM_FEATS, cat=CAT_FEATS):
    return ColumnTransformer([
        ("num", StandardScaler() if scale else "passthrough", num),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False), cat)])

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
""")

md("## 7. Model development and hyper-parameter tuning\n- **Baseline:** `DummyClassifier` (predicts the training default rate for everyone)\n- **Logistic Regression** (transparent industry benchmark)\n- **Random Forest** (bagging ensemble)\n- **XGBoost** (gradient boosting)\n\nTuning: `RandomizedSearchCV`, 5-fold stratified CV on the training set, scoring = **PR-AUC (average precision)**, which focuses on the minority (default) class.")
code(r"""
search_spaces = {
    "Logistic Regression": (
        Pipeline([("pre", make_pre(True)), ("clf", LogisticRegression(max_iter=3000, solver="lbfgs"))]),
        {"clf__C": loguniform(1e-3, 1e2)}, 12),
    "Random Forest": (
        Pipeline([("pre", make_pre(False)), ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1))]),
        {"clf__n_estimators": randint(200, 401), "clf__max_depth": randint(4, 16),
         "clf__min_samples_leaf": randint(5, 60), "clf__max_features": ["sqrt", "log2", 0.3]}, 12),
    "XGBoost": (
        Pipeline([("pre", make_pre(False)), ("clf", XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss", tree_method="hist"))]),
        {"clf__n_estimators": randint(150, 501), "clf__max_depth": randint(2, 7),
         "clf__learning_rate": loguniform(0.01, 0.2), "clf__subsample": uniform(0.6, 0.4),
         "clf__colsample_bytree": uniform(0.5, 0.5), "clf__min_child_weight": randint(1, 11),
         "clf__reg_lambda": loguniform(0.1, 10)}, 20),
}

models, tuning_rows = {}, []
for name, (pipe, space, n_iter) in search_spaces.items():
    t0 = time.time()
    rs = RandomizedSearchCV(pipe, space, n_iter=n_iter, scoring="average_precision", cv=CV,
                            random_state=SEED, n_jobs=1, refit=True)
    rs.fit(X_tr, y_tr)
    models[name] = rs.best_estimator_
    best = {k.replace("clf__", ""): (round(v, 4) if isinstance(v, float) else v) for k, v in rs.best_params_.items()}
    tuning_rows.append({"Model": name, "Iterations": n_iter, "CV PR-AUC (mean)": rs.best_score_,
                        "CV PR-AUC (sd)": rs.cv_results_["std_test_score"][rs.best_index_],
                        "Best hyper-parameters": json.dumps(best), "Time (s)": round(time.time() - t0, 1)})
    print(f"{name:<20} CV PR-AUC = {rs.best_score_:.4f}   {best}")

models["Baseline (Dummy)"] = Pipeline([("pre", make_pre(True)), ("clf", DummyClassifier(strategy="prior"))]).fit(X_tr, y_tr)
tuning = pd.DataFrame(tuning_rows)
tuning.to_csv("results/table_tuning.csv", index=False)
tuning
""")
code(r"""
BEST_PARAMS = {r["Model"]: json.loads(r["Best hyper-parameters"]) for r in tuning_rows}
ORDER = ["Baseline (Dummy)", "Logistic Regression", "Random Forest", "XGBoost"]
""")

md("## 8. Test-set evaluation\nThe decision threshold for each model is chosen on the **validation** set to maximise F1 for the default class. That threshold is then applied unchanged to the **test** set.")
code(r"""
def best_f1_threshold(y, p):
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f1s = [metrics.f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])

def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1); e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any(): e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e

def evaluate(y, p, t):
    yhat = (p >= t).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    return {"ROC-AUC": metrics.roc_auc_score(y, p), "PR-AUC": metrics.average_precision_score(y, p),
            "F1": metrics.f1_score(y, yhat, zero_division=0), "Precision": metrics.precision_score(y, yhat, zero_division=0),
            "Recall": metrics.recall_score(y, yhat), "Balanced acc.": metrics.balanced_accuracy_score(y, yhat),
            "Accuracy": metrics.accuracy_score(y, yhat), "Brier": metrics.brier_score_loss(y, p), "ECE": ece(np.asarray(y), p),
            "Threshold": t, "TP": tp, "FP": fp, "FN": fn, "TN": tn}

P_VA = {m: models[m].predict_proba(X_va)[:, 1] for m in ORDER}
P_TE = {m: models[m].predict_proba(X_te)[:, 1] for m in ORDER}
THR = {m: (0.5 if m.startswith("Baseline") else best_f1_threshold(y_va, P_VA[m])) for m in ORDER}

perf = pd.DataFrame({m: evaluate(y_te, P_TE[m], THR[m]) for m in ORDER}).T
perf_default05 = pd.DataFrame({m: evaluate(y_te, P_TE[m], 0.5) for m in ORDER}).T[["F1", "Precision", "Recall"]]
perf.to_csv("results/table_test_performance.csv")
perf
""")
code(r"""
print("Same models at the naive 0.5 threshold (for comparison):")
perf_default05
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for m in ORDER:
    fpr, tpr, _ = metrics.roc_curve(y_te, P_TE[m]); axes[0].plot(fpr, tpr, label=f"{m} (AUC={perf.loc[m,'ROC-AUC']:.3f})")
    pr, rc, _ = metrics.precision_recall_curve(y_te, P_TE[m]); axes[1].plot(rc, pr, label=f"{m} (AP={perf.loc[m,'PR-AUC']:.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=.8); axes[0].set(xlabel="False positive rate", ylabel="True positive rate (recall)", title="ROC curves (test)")
axes[1].axhline(y_te.mean(), color="k", ls="--", lw=.8); axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-recall curves (test)")
for a in axes: a.legend(fontsize=8)
plt.tight_layout(); plt.savefig("figures/fig4_roc_pr.png"); plt.show()
""")
code(r"""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, m in zip(axes, ORDER[1:]):
    cm = metrics.confusion_matrix(y_te, (P_TE[m] >= THR[m]).astype(int), labels=[1, 0])
    metrics.ConfusionMatrixDisplay(cm, display_labels=["Default", "No default"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{m} (threshold {THR[m]:.2f})")
plt.tight_layout(); plt.savefig("figures/fig5_confusion_matrices.png"); plt.show()
""")

md("## 9. Calibration\nA calibrated model's predicted probability matches the observed default rate (a predicted 30% PD should default about 30% of the time). We report the **Brier score** and **Expected Calibration Error (ECE)** and draw reliability curves. We also test (a) what **class re-weighting** (a common imbalance fix) does to calibration and (b) whether **isotonic re-calibration** on the validation set repairs it.")
code(r"""
# (a) Imbalance ablation: XGBoost re-trained with class weighting (scale_pos_weight = n_neg / n_pos)
spw = (y_tr == 0).sum() / (y_tr == 1).sum()
xgb_w = Pipeline([("pre", make_pre(False)), ("clf", XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss",
                   tree_method="hist", scale_pos_weight=spw, **BEST_PARAMS["XGBoost"]))]).fit(X_tr, y_tr)
P_VA["XGBoost (class-weighted)"] = xgb_w.predict_proba(X_va)[:, 1]
P_TE["XGBoost (class-weighted)"] = xgb_w.predict_proba(X_te)[:, 1]

# (b) Isotonic re-calibration fitted on the validation set (the model itself is frozen)
for base, model in [("XGBoost", models["XGBoost"]), ("XGBoost (class-weighted)", xgb_w)]:
    cal = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic").fit(X_va, y_va)
    P_TE[base + " + isotonic"] = cal.predict_proba(X_te)[:, 1]

cal_names = ["Logistic Regression", "Random Forest", "XGBoost", "XGBoost (class-weighted)", "XGBoost (class-weighted) + isotonic", "XGBoost + isotonic"]
calib = pd.DataFrame({m: {"Brier": metrics.brier_score_loss(y_te, P_TE[m]), "ECE": ece(np.asarray(y_te), P_TE[m]),
                          "Mean predicted PD": P_TE[m].mean(), "Observed default rate": y_te.mean(),
                          "ROC-AUC": metrics.roc_auc_score(y_te, P_TE[m])} for m in cal_names}).T
calib.to_csv("results/table_calibration.csv")
calib
""")
code(r"""
fig, ax = plt.subplots(figsize=(6.5, 6))
for m in ["Logistic Regression", "Random Forest", "XGBoost", "XGBoost (class-weighted)", "XGBoost (class-weighted) + isotonic"]:
    fp_, mp_ = calibration_curve(y_te, P_TE[m], n_bins=10, strategy="quantile")
    ax.plot(mp_, fp_, marker="o", ms=4, label=m)
ax.plot([0, 1], [0, 1], "k--", lw=.8, label="Perfect calibration")
ax.set(xlabel="Mean predicted probability of default", ylabel="Observed default rate", title="Reliability curves (test set)")
ax.legend(fontsize=8); plt.savefig("figures/fig6_reliability.png"); plt.show()
""")

md("## 10. Subgroup error rates and fairness\nFor each demographic group we report, on the test set, using the best discriminating model:\n- **Approval rate**: share predicted *no default* (the favourable outcome)\n- **FPR (wrongful rejection)**: share of good borrowers (no default) wrongly flagged as defaulters\n- **FNR (missed default)**: share of real defaulters wrongly approved\n- **Group AUC** and **group ECE** (does the model rank and calibrate equally well in each group?)\n\nGap metrics (largest minus smallest across groups):\n- **Statistical parity difference (SPD)**: gap in approval rates\n- **Equal opportunity difference (EOD)**: gap in FPR, i.e. in the chance that a *good* borrower is wrongly rejected\n- **Equalised odds difference (EqOdds)**: the larger of the FPR gap and the FNR gap\n- **Disparate impact ratio (DIR)**: lowest approval rate divided by highest (the US 'four-fifths rule' flags values below 0.80)")
code(r"""
BEST = perf.loc[["Logistic Regression", "Random Forest", "XGBoost"], "ROC-AUC"].idxmax()
print("Best discriminating model on test ROC-AUC:", BEST)

def group_table(frame, y, p, t, min_n=30):
    yhat = (p >= t).astype(int); rows = []
    for name, col in SENSITIVE.items():
        for g in frame[col].dropna().unique():
            m = (frame[col] == g).values
            if m.sum() < min_n: continue
            yy, pp, hh = np.asarray(y)[m], p[m], yhat[m]
            neg, pos = yy == 0, yy == 1
            rows.append({"Attribute": name, "Group": str(g), "n": int(m.sum()), "Observed default rate": yy.mean(),
                         "Mean predicted PD": pp.mean(), "Approval rate": 1 - hh.mean(),
                         "FPR (good wrongly rejected)": hh[neg].mean(), "FNR (default missed)": 1 - hh[pos].mean(),
                         "Precision": metrics.precision_score(yy, hh, zero_division=0),
                         "AUC": metrics.roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else np.nan, "ECE": ece(yy, pp)})
    return pd.DataFrame(rows).sort_values(["Attribute", "Group"]).reset_index(drop=True)

def gap_table(gt):
    out = []
    for a, g in gt.groupby("Attribute", sort=False):
        out.append({"Attribute": a, "SPD (approval gap)": g["Approval rate"].max() - g["Approval rate"].min(),
                    "DIR (min/max approval)": g["Approval rate"].min() / g["Approval rate"].max(),
                    "EOD (FPR gap)": g["FPR (good wrongly rejected)"].max() - g["FPR (good wrongly rejected)"].min(),
                    "FNR gap": g["FNR (default missed)"].max() - g["FNR (default missed)"].min(),
                    "EqOdds": max(g["FPR (good wrongly rejected)"].max() - g["FPR (good wrongly rejected)"].min(),
                                  g["FNR (default missed)"].max() - g["FNR (default missed)"].min()),
                    "AUC gap": g["AUC"].max() - g["AUC"].min(), "ECE gap": g["ECE"].max() - g["ECE"].min(),
                    "Base-rate gap": g["Observed default rate"].max() - g["Observed default rate"].min()})
    return pd.DataFrame(out)

grp_best = group_table(test, y_te, P_TE[BEST], THR[BEST])
grp_best.to_csv("results/table_subgroup_metrics_best.csv", index=False)
grp_best
""")
code(r"""
gaps_all = []
for m in ["Logistic Regression", "Random Forest", "XGBoost"]:
    g = gap_table(group_table(test, y_te, P_TE[m], THR[m])); g.insert(0, "Model", m); gaps_all.append(g)
gaps_all = pd.concat(gaps_all, ignore_index=True)
gaps_all.to_csv("results/table_fairness_gaps_by_model.csv", index=False)
gaps_all
""")
code(r"""
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
for ax, name in zip(axes, SENSITIVE):
    g = grp_best[grp_best.Attribute == name].set_index("Group")
    g[["FPR (good wrongly rejected)", "FNR (default missed)"]].plot.bar(ax=ax, color=["#DD8452", "#4C72B0"], legend=(name == "Sex"))
    ax.set_title(f"{BEST}: error rates by {name.lower()}"); ax.tick_params(axis="x", rotation=30); ax.set_xlabel("")
axes[0].set_ylabel("Rate")
plt.tight_layout(); plt.savefig("figures/fig7_subgroup_error_rates.png"); plt.show()
""")
code(r"""
# 95% bootstrap confidence intervals (1,000 resamples of the test set) for the sex gaps of the best model
rng = np.random.default_rng(SEED); p_b = P_TE[BEST]; yv = np.asarray(y_te); sexv = test["SEX_LBL"].values
def sex_gaps(idx):
    yh = (p_b[idx] >= THR[BEST]).astype(int); yy = yv[idx]; s = sexv[idx]; r = {}
    for lab in ["Female", "Male"]:
        m = s == lab
        r[lab] = (1 - yh[m].mean(), yh[m & (yy == 0)].sum() / max((m & (yy == 0)).sum(), 1),
                  1 - yh[m & (yy == 1)].sum() / max((m & (yy == 1)).sum(), 1))
    return [r["Female"][i] - r["Male"][i] for i in range(3)]
boot = np.array([sex_gaps(rng.integers(0, len(yv), len(yv))) for _ in range(1000)])
point = sex_gaps(np.arange(len(yv)))
sex_ci = pd.DataFrame({"Female minus male": ["Approval rate", "FPR (good wrongly rejected)", "FNR (default missed)"],
                       "Point estimate": point, "95% CI low": np.percentile(boot, 2.5, axis=0), "95% CI high": np.percentile(boot, 97.5, axis=0)})
sex_ci.to_csv("results/table_sex_gap_bootstrap_ci.csv", index=False)
sex_ci
""")

md("## 11. Fairness mitigation\n**(a) Fairness through unawareness** (as tested in base paper 1): re-train every model **without** SEX, AGE and MARRIAGE and compare accuracy and gaps.\n\n**(b) Threshold optimisation** (extends base paper 3): keep the model fixed and only change the decision cut-off(s).\n1. *Single global threshold* chosen on validation by maximising J(ω) = ω · balanced accuracy − (1 − ω) · |EOD across sex|, for ω = 1, 0.75, 0.5, 0.25.\n2. *Group-specific thresholds* for sex chosen on validation so that good borrowers of both sexes face the same wrongful-rejection rate (equal FPR), keeping the overall FPR at the level of the F1-optimal threshold.")
code(r"""
SENS_DROP = ["SEX", "AGE", "MARRIAGE"]
NUM_U = [c for c in NUM_FEATS if c not in SENS_DROP]; CAT_U = [c for c in CAT_FEATS if c not in SENS_DROP]
make_clf = {"Logistic Regression": lambda: LogisticRegression(max_iter=3000, **BEST_PARAMS["Logistic Regression"]),
            "Random Forest": lambda: RandomForestClassifier(random_state=SEED, n_jobs=-1, **BEST_PARAMS["Random Forest"]),
            "XGBoost": lambda: XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss", tree_method="hist", **BEST_PARAMS["XGBoost"])}
unaware_rows = []
for m, mk in make_clf.items():
    pipe = Pipeline([("pre", make_pre(m == "Logistic Regression", NUM_U, CAT_U)), ("clf", mk())]).fit(train[NUM_U + CAT_U], y_tr)
    pv, pt = pipe.predict_proba(val[NUM_U + CAT_U])[:, 1], pipe.predict_proba(test[NUM_U + CAT_U])[:, 1]
    t = best_f1_threshold(y_va, pv); P_TE[m + " (unaware)"] = pt
    for version, p_, t_ in [("With demographics", P_TE[m], THR[m]), ("Unaware (no SEX/AGE/MARRIAGE)", pt, t)]:
        ev = evaluate(y_te, p_, t_); gp = gap_table(group_table(test, y_te, p_, t_)).set_index("Attribute")
        unaware_rows.append({"Model": m, "Version": version, "ROC-AUC": ev["ROC-AUC"], "PR-AUC": ev["PR-AUC"], "F1": ev["F1"],
                             "EOD sex": gp.loc["Sex", "EOD (FPR gap)"], "SPD sex": gp.loc["Sex", "SPD (approval gap)"],
                             "EOD age": gp.loc["Age band", "EOD (FPR gap)"], "EOD marital": gp.loc["Marital status", "EOD (FPR gap)"]})
unaware = pd.DataFrame(unaware_rows)
unaware.to_csv("results/table_unawareness.csv", index=False)
unaware
""")
code(r"""
sex_va, sex_te = val["SEX_LBL"].values, test["SEX_LBL"].values
def fair_stats(y, p, s, thr):   # thr: float or dict {group: threshold}
    t = np.array([thr[g] for g in s]) if isinstance(thr, dict) else np.full(len(p), thr)
    yh = (p >= t).astype(int); y = np.asarray(y); r = {}
    for g in ["Female", "Male"]:
        m = s == g
        r[g] = {"fpr": yh[m & (y == 0)].mean(), "fnr": 1 - yh[m & (y == 1)].mean(), "appr": 1 - yh[m].mean()}
    return {"Balanced acc.": metrics.balanced_accuracy_score(y, yh), "F1": metrics.f1_score(y, yh),
            "Recall": metrics.recall_score(y, yh), "Precision": metrics.precision_score(y, yh),
            "EOD sex (FPR gap)": abs(r["Female"]["fpr"] - r["Male"]["fpr"]), "FNR gap sex": abs(r["Female"]["fnr"] - r["Male"]["fnr"]),
            "SPD sex (approval gap)": abs(r["Female"]["appr"] - r["Male"]["appr"]), "Overall FPR": yh[y == 0].mean()}

grid = np.round(np.arange(0.05, 0.951, 0.01), 2); pv, pt = P_VA[BEST], P_TE[BEST]
sweep = pd.DataFrame([{"Threshold": t, **fair_stats(y_va, pv, sex_va, t)} for t in grid])
th_rows = [{"Policy": "Default 0.50 threshold", "Threshold(s)": "0.50", **fair_stats(y_te, pt, sex_te, 0.5)},
           {"Policy": "F1-optimal threshold", "Threshold(s)": f"{THR[BEST]:.2f}", **fair_stats(y_te, pt, sex_te, THR[BEST])}]
for w in [1.0, 0.75, 0.5, 0.25]:
    J = w * sweep["Balanced acc."] - (1 - w) * sweep["EOD sex (FPR gap)"]
    t = float(sweep.loc[J.idxmax(), "Threshold"])
    th_rows.append({"Policy": f"Scalarised J, omega = {w}", "Threshold(s)": f"{t:.2f}", **fair_stats(y_te, pt, sex_te, t)})

# Group-specific thresholds: equalise FPR across sexes at the overall FPR of the F1-optimal threshold (fitted on validation)
target_fpr = fair_stats(y_va, pv, sex_va, THR[BEST])["Overall FPR"]; gthr = {}
for g in ["Female", "Male"]:
    neg = (sex_va == g) & (np.asarray(y_va) == 0)
    gthr[g] = float(np.quantile(pv[neg], 1 - target_fpr))
th_rows.append({"Policy": "Group thresholds (equal FPR by sex)", "Threshold(s)": f"F {gthr['Female']:.3f} / M {gthr['Male']:.3f}",
                **fair_stats(y_te, pt, sex_te, gthr)})
thresholds = pd.DataFrame(th_rows)
thresholds.to_csv("results/table_threshold_policies.csv", index=False)
thresholds
""")
code(r"""
fig, ax1 = plt.subplots(figsize=(8, 4.5))
ax1.plot(sweep["Threshold"], sweep["Balanced acc."], label="Balanced accuracy", color="#4C72B0")
ax1.plot(sweep["Threshold"], sweep["F1"], label="F1", color="#55A868")
ax1.set_xlabel("Decision threshold (validation set)"); ax1.set_ylabel("Performance")
ax2 = ax1.twinx()
ax2.plot(sweep["Threshold"], sweep["EOD sex (FPR gap)"], color="#C44E52", ls="--", label="EOD sex (FPR gap)")
ax2.plot(sweep["Threshold"], sweep["SPD sex (approval gap)"], color="#8172B2", ls=":", label="SPD sex (approval gap)")
ax2.set_ylabel("Fairness gap (lower is fairer)")
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
plt.title(f"{BEST}: performance-fairness trade-off across thresholds")
plt.savefig("figures/fig8_threshold_tradeoff.png"); plt.show()
""")

md("## 12. Explainability with SHAP\nSHAP (SHapley Additive exPlanations) splits each prediction into contributions from each feature. **Global** view: which features matter most overall. **Local** view: why one particular borrower received their score. SHAP is computed for the tuned XGBoost model on the test set (values are in log-odds units).")
code(r"""
xgb_pipe = models["XGBoost"]
pre_fit = xgb_pipe.named_steps["pre"]
feat_names = [n.split("__", 1)[1] for n in pre_fit.get_feature_names_out()]
X_te_t = pd.DataFrame(pre_fit.transform(X_te), columns=feat_names)
explainer = shap.TreeExplainer(xgb_pipe.named_steps["clf"])
sv = explainer(X_te_t)

shap_imp = pd.DataFrame({"Feature": feat_names, "Mean |SHAP|": np.abs(sv.values).mean(axis=0)}).sort_values("Mean |SHAP|", ascending=False)
shap_imp["Share of total"] = shap_imp["Mean |SHAP|"] / shap_imp["Mean |SHAP|"].sum()
shap_imp["Rank"] = range(1, len(shap_imp) + 1)
shap_imp.to_csv("results/table_shap_importance.csv", index=False)
shap_imp.head(12)
""")
code(r"""
demo_cols = [c for c in feat_names if c.startswith(("SEX", "AGE", "MARRIAGE", "EDUCATION"))]
print("Demographic features and their SHAP rank:")
print(shap_imp[shap_imp.Feature.isin(demo_cols)][["Feature", "Mean |SHAP|", "Share of total", "Rank"]].to_string(index=False))
print(f"\nCombined share of all demographic features: {shap_imp[shap_imp.Feature.isin(demo_cols)]['Share of total'].sum():.2%}")
""")
code(r"""
shap.plots.beeswarm(sv, max_display=15, show=False)
plt.title("SHAP summary (XGBoost, test set)"); plt.savefig("figures/fig9_shap_beeswarm.png"); plt.show()
shap.plots.bar(sv, max_display=15, show=False)
plt.title("Mean |SHAP| (global importance)"); plt.savefig("figures/fig10_shap_bar.png"); plt.show()
""")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, f in zip(axes, ["PAY_1", "LIMIT_BAL"]):
    j = feat_names.index(f); ax.scatter(X_te_t[f], sv.values[:, j], s=4, alpha=.4)
    ax.axhline(0, color="k", lw=.6); ax.set(xlabel=f, ylabel="SHAP value (log-odds)", title=f"SHAP dependence: {f}")
axes[1].set_xscale("log")
plt.tight_layout(); plt.savefig("figures/fig11_shap_dependence.png"); plt.show()
""")
code(r"""
# Local explanations: one rejected borrower who really defaulted, and one approved borrower close to the cut-off
p_x = P_TE["XGBoost"]; t_x = THR["XGBoost"]
i_rej = int(np.where((p_x >= t_x) & (y_te.values == 1))[0][np.argmax(p_x[(p_x >= t_x) & (y_te.values == 1)])])
cand = np.where((p_x < t_x) & (p_x > t_x - 0.05))[0]; i_near = int(cand[0])
for i, lab, fn in [(i_rej, "High-risk borrower (rejected, actually defaulted)", "fig12_shap_local_rejected.png"),
                   (i_near, "Borderline borrower (approved, just below the cut-off)", "fig13_shap_local_borderline.png")]:
    print(f"{lab}: index {i}, predicted PD = {p_x[i]:.3f}, actual = {y_te.values[i]}")
    shap.plots.waterfall(sv[i], max_display=10, show=False); plt.title(lab); plt.savefig("figures/" + fn); plt.show()
""")
code(r"""
# Does the model's reliance on the SEX variable differ by sex? (mean SHAP of SEX feature within each group)
sex_feat = [c for c in feat_names if c.startswith("SEX")][0]
j = feat_names.index(sex_feat)
pd.DataFrame({"Group": test["SEX_LBL"], "SHAP of SEX feature": sv.values[:, j]}).groupby("Group").agg(["mean", "std"])
""")

md("## 13. Robustness across random seeds\nTo avoid reporting one lucky split, the full train/validation/test split is repeated with **five seeds** (0 to 4). Models are re-fitted with the tuned hyper-parameters, thresholds are re-chosen on each validation set, and results are summarised as mean ± standard deviation.")
code(r"""
rob = []
for s in range(5):
    tr_, va_, te_ = make_split(df, s)
    for m, mk in make_clf.items():
        clf = mk(); clf.set_params(random_state=s) if hasattr(clf, "random_state") and m != "Logistic Regression" else None
        pipe = Pipeline([("pre", make_pre(m == "Logistic Regression")), ("clf", clf)]).fit(tr_[ALL_FEATS], tr_[TARGET])
        pv_, pt_ = pipe.predict_proba(va_[ALL_FEATS])[:, 1], pipe.predict_proba(te_[ALL_FEATS])[:, 1]
        t_ = best_f1_threshold(va_[TARGET], pv_); ev = evaluate(te_[TARGET], pt_, t_)
        gp = gap_table(group_table(te_, te_[TARGET], pt_, t_)).set_index("Attribute")
        rob.append({"Seed": s, "Model": m, "ROC-AUC": ev["ROC-AUC"], "PR-AUC": ev["PR-AUC"], "F1": ev["F1"],
                    "Recall": ev["Recall"], "Brier": ev["Brier"], "ECE": ev["ECE"],
                    "EOD sex": gp.loc["Sex", "EOD (FPR gap)"], "EOD age": gp.loc["Age band", "EOD (FPR gap)"],
                    "EOD education": gp.loc["Education", "EOD (FPR gap)"], "DIR education": gp.loc["Education", "DIR (min/max approval)"]})
rob = pd.DataFrame(rob); rob.to_csv("results/table_robustness_raw.csv", index=False)
rob_summary = rob.drop(columns="Seed").groupby("Model").agg(["mean", "std"]).reindex(["Logistic Regression", "Random Forest", "XGBoost"])
rob_summary.to_csv("results/table_robustness_summary.csv")
rob_summary
""")

md("## 14. Business analysis: expected cost of lending decisions\nMissing a defaulter (false negative) usually costs a lender much more than wrongly rejecting a good client (false positive: lost interest income). We **assume** a cost ratio FN : FP = 5 : 1 for the main analysis and test ratios from 2 to 10 for sensitivity. The cost-minimising threshold is chosen on the validation set and evaluated on the test set.")
code(r"""
def exp_cost(y, p, t, c_fn, c_fp=1.0):
    yh = (p >= t).astype(int); y = np.asarray(y)
    return (c_fn * ((yh == 0) & (y == 1)).sum() + c_fp * ((yh == 1) & (y == 0)).sum()) / len(y)

cost_rows = []
for ratio in [2, 3, 5, 7, 10]:
    for m in ["Logistic Regression", "Random Forest", "XGBoost"]:
        costs = [exp_cost(y_va, P_VA[m], t, ratio) for t in grid]; t_c = float(grid[int(np.argmin(costs))])
        base_cost = exp_cost(y_te, np.zeros(len(y_te)), 0.5, ratio)   # approve everyone
        c05, cbest = exp_cost(y_te, P_TE[m], 0.5, ratio), exp_cost(y_te, P_TE[m], t_c, ratio)
        cost_rows.append({"FN:FP cost ratio": ratio, "Model": m, "Cost-optimal threshold": t_c,
                          "Cost per applicant: approve all": base_cost, "Cost: threshold 0.5": c05,
                          "Cost: cost-optimal threshold": cbest, "Saving vs approve-all": 1 - cbest / base_cost,
                          "Saving vs 0.5 threshold": 1 - cbest / c05})
cost = pd.DataFrame(cost_rows); cost.to_csv("results/table_cost_analysis.csv", index=False)
cost[cost["FN:FP cost ratio"] == 5]
""")
code(r"""
piv = cost.pivot(index="FN:FP cost ratio", columns="Model", values="Cost-optimal threshold")
piv.plot(marker="o", figsize=(7, 4)); plt.ylabel("Cost-minimising threshold"); plt.title("Optimal cut-off falls as missed defaults become costlier")
plt.savefig("figures/fig14_cost_threshold.png"); plt.show()
piv
""")

md("## 15. Summary of key results\nThe cell below prints the main numbers used in the article, so every figure in the manuscript can be traced to this notebook.")
code(r"""
summary = {
    "rows_after_cleaning": len(df), "duplicates_removed": int(n_dup), "default_rate": round(df[TARGET].mean(), 4),
    "best_model_test_auc": BEST,
    "test_metrics": perf.loc[ORDER, ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Brier", "ECE", "Threshold"]].round(4).to_dict(orient="index"),
    "calibration": calib[["Brier", "ECE", "Mean predicted PD"]].round(4).to_dict(orient="index"),
    "fairness_gaps_best": gap_table(grp_best).round(4).to_dict(orient="records"),
    "demographic_shap_share": round(float(shap_imp[shap_imp.Feature.isin(demo_cols)]["Share of total"].sum()), 4),
    "top5_shap": shap_imp.head(5)["Feature"].tolist(),
}
with open("results/summary.json", "w") as f: json.dump(summary, f, indent=2, default=str)
print(json.dumps(summary, indent=2, default=str)[:6000])
""")

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "GP9_Credit_Fairness_ML.ipynb")
print("written", len(cells), "cells")
