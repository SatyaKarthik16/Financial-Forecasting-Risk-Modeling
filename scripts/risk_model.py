"""
Credit risk modeling pipeline with imbalance handling and model comparison.
"""

import argparse
import os
import sys
import warnings

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

FEATURE_COLS = [
    "Amount", "AmountLog", "CreditScore", "IsHighValue", "DayOfWeek",
    "PaymentType_Debit Card", "PaymentType_PayPal", "PaymentType_Wire Transfer",
]
TARGET_COL = "Defaulted"
RANDOM_STATE = 42
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)


def load_features(filepath: str):
    sys.path.insert(0, os.path.dirname(__file__))
    from etl_pipeline import run_etl

    df = run_etl(filepath)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0
    return df[FEATURE_COLS], df[TARGET_COL]


def build_pipelines():
    return {
        "Logistic Regression": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "XGBoost": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, scale_pos_weight=9, eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0)),
        ]),
    }


def main(filepath: str):
    X, y = load_features(filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    pipelines = build_pipelines()
    results = []

    for name, pipe in pipelines.items():
        print(f"\n[Risk Model] Training {name}...")
        cv = cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5), scoring="roc_auc", n_jobs=-1)
        print(f"  5-Fold CV ROC-AUC: {cv.mean():.4f} +/- {cv.std():.4f}")

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)

        print(f"  ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Default", "Default"], yticklabels=["No Default", "Default"])
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.savefig(f"images/confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=140)
        plt.close()

        results.append({"Model": name, "ROC_AUC": round(roc_auc, 4), "PR_AUC": round(pr_auc, 4), "pipeline": pipe})

    report_df = pd.DataFrame([{k: v for k, v in r.items() if k != "pipeline"} for r in results])
    report_df.to_csv("credit_risk_model_report.csv", index=False)
    best = max(results, key=lambda x: x["ROC_AUC"])
    joblib.dump(best["pipeline"], "models/credit_risk_model.pkl")
    print(f"\nBest model: {best['Model']} | ROC-AUC={best['ROC_AUC']}")
    print("Saved models/credit_risk_model.pkl and credit_risk_model_report.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic_transactions.csv")
    args = p.parse_args()
    main(args.data)
