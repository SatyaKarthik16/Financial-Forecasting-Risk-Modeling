"""
Credit risk modeling pipeline with imbalance handling and model comparison.
"""

import argparse
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
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
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(n_estimators=350, max_depth=10, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "XGBoost": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            (
                "clf",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    verbosity=0,
                ),
            ),
        ]),
    }


def save_confusion_matrix_panel(results):
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        sns.heatmap(
            res["cm"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Default", "Default"],
            yticklabels=["No Default", "Default"],
            ax=ax,
        )
        ax.set_title(res["Model"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig("images/confusion_matrices.png", dpi=150)
    plt.close()


def save_roc_pr_curves(results, y_positive_rate: float):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_roc, ax_pr = axes

    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="No-skill")
    ax_pr.hlines(y_positive_rate, 0, 1, linestyles="--", colors="gray", label="No-skill")

    for res in results:
        fpr, tpr = res["roc_curve"]
        recall, precision = res["pr_curve"]
        ax_roc.plot(fpr, tpr, linewidth=2, label=f"{res['Model']} (AUC={res['ROC_AUC']:.3f})")
        ax_pr.plot(recall, precision, linewidth=2, label=f"{res['Model']} (AP={res['PR_AUC']:.3f})")

    ax_roc.set_title("ROC Curves")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(loc="lower right")

    ax_pr.set_title("Precision-Recall Curves")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("images/roc_pr_curves.png", dpi=150)
    plt.close()


def save_model_explainability(best_pipeline, x_sample: pd.DataFrame):
    model = best_pipeline.named_steps["clf"]

    # Prefer SHAP for tree models; fall back to model-native importance for linear/non-tree models.
    if isinstance(model, XGBClassifier):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, x_sample, plot_type="bar", show=False, max_display=12)
            plt.tight_layout()
            plt.savefig("images/shap_feature_importance.png", dpi=150)
            plt.close()
            return "SHAP"
        except Exception as exc:
            print(f"[Risk Model] SHAP explainability failed, using fallback importance: {exc}")

    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_).ravel()
        feature_names = list(x_sample.columns)
        title = "Model Explainability (|Coefficients|)"
    elif hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_)
        feature_names = list(x_sample.columns)
        title = "Model Explainability (Feature Importances)"
    else:
        print("[Risk Model] Explainability skipped: model has no importances or coefficients.")
        return "Skipped"

    imp_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    imp_df = imp_df.sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", color="#4c78a8")
    plt.title(title)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("images/shap_feature_importance.png", dpi=150)
    plt.close()
    return "Fallback"


def save_business_impact(best_result):
    tn, fp, fn, tp = best_result["cm"].ravel()
    loss_per_fn = 2500.0
    review_cost_per_fp = 150.0
    intervention_gain_per_tp = 700.0

    baseline_loss = (tp + fn) * loss_per_fn
    model_loss = (fn * loss_per_fn) + (fp * review_cost_per_fp) - (tp * intervention_gain_per_tp)
    savings = baseline_loss - model_loss

    labels = ["Naive Baseline Cost", "Model Residual Cost", "Estimated Savings"]
    values = [baseline_loss, max(model_loss, 0), max(savings, 0)]
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]

    plt.figure(figsize=(9, 5))
    sns.barplot(x=labels, y=values, palette=colors)
    plt.title("Estimated Business Impact (Illustrative)")
    plt.ylabel("USD")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("images/business_impact.png", dpi=150)
    plt.close()


def main(filepath: str):
    X, y = load_features(filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    pipelines = build_pipelines()
    results = []
    cv_splitter = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)

    for name, pipe in pipelines.items():
        print(f"\n[Risk Model] Training {name}...")
        cv = cross_val_score(pipe, X_train, y_train, cv=cv_splitter, scoring="roc_auc", n_jobs=-1)
        print(f"  5-Fold CV ROC-AUC: {cv.mean():.4f} +/- {cv.std():.4f}")

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)

        print(f"  ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

        cm = confusion_matrix(y_test, y_pred)
        results.append(
            {
                "Model": name,
                "ROC_AUC": float(roc_auc),
                "PR_AUC": float(pr_auc),
                "CV_ROC_AUC_MEAN": float(cv.mean()),
                "CV_ROC_AUC_STD": float(cv.std()),
                "cm": cm,
                "roc_curve": (fpr, tpr),
                "pr_curve": (recall, precision),
                "pipeline": pipe,
            }
        )

    save_confusion_matrix_panel(results)
    save_roc_pr_curves(results, y_positive_rate=float(y_test.mean()))

    report_df = pd.DataFrame(
        [
            {
                "Model": r["Model"],
                "ROC_AUC": round(r["ROC_AUC"], 4),
                "PR_AUC": round(r["PR_AUC"], 4),
                "CV_ROC_AUC_MEAN": round(r["CV_ROC_AUC_MEAN"], 4),
                "CV_ROC_AUC_STD": round(r["CV_ROC_AUC_STD"], 4),
            }
            for r in results
        ]
    )
    report_df.to_csv("credit_risk_model_report.csv", index=False)
    best = max(results, key=lambda x: x["ROC_AUC"])
    joblib.dump(best["pipeline"], "models/credit_risk_model.pkl")
    sample_n = min(2000, len(X_train))
    x_sample = X_train.sample(sample_n, random_state=RANDOM_STATE)
    explainability_mode = save_model_explainability(best["pipeline"], x_sample)
    save_business_impact(best)

    print(f"\nBest model: {best['Model']} | ROC-AUC={best['ROC_AUC']:.4f}")
    print(
        "Saved models/credit_risk_model.pkl, credit_risk_model_report.csv, "
        "images/confusion_matrices.png, images/roc_pr_curves.png, "
        f"images/shap_feature_importance.png ({explainability_mode}), and images/business_impact.png"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synthetic_transactions.csv")
    args = p.parse_args()
    main(args.data)
