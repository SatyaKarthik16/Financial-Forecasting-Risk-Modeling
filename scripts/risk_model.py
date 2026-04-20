import argparse, os, sys, warnings
import joblib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
warnings.filterwarnings("ignore")

FEATURE_COLS = ["Amount","AmountLog","CreditScore","IsHighValue","DayOfWeek",
    "PaymentType_Debit Card","PaymentType_PayPal","PaymentType_Wire Transfer"]
TARGET_COL = "Defaulted"
RANDOM_STATE = 42
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

def load_features(filepath):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")),
        ]),
        "Random Forest": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "XGBoost": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                scale_pos_weight=9, eval_metric="logloss", random_state=RANDOM_STATE, verbosity=0)),
        ]),
    }

def main(filepath):
    print("\n[Risk Model] Loading data...")
    X, y = load_features(filepath)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    pipelines = build_pipelines()
    results = []
    for name, pipeline in pipelines.items():
        print(f"\n[Risk Model] Training {name}...")
        cv = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(5), scoring="roc_auc", n_jobs=-1)
        print(f"  5-Fold CV ROC-AUC: {cv.mean():.4f} +/- {cv.std():.4f}")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:,1]
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc  = average_precision_score(y_test, y_prob)
        print(f"  ROC-AUC={roc_auc:.4f}  PR-AUC={pr_auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Default","Default"]))
        results.append({"name":name,"pipeline":pipeline,"roc_auc":roc_auc,"pr_auc":pr_auc,"y_pred":y_pred,"y_prob":y_prob,"y_test":y_test})
    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\n[Risk Model] Best: {best['name']} ROC-AUC={best['roc_auc']:.4f}")
    joblib.dump(best["pipeline"], "models/credit_risk_model.pkl")
    pd.DataFrame([{"Model":r["name"],"ROC_AUC":round(r["roc_auc"],4),"PR_AUC":round(r["pr_auc"],4)} for r in results]).to_csv("credit_risk_model_report.csv", index=False)
    print("[Risk Model] Saved model and report.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synthetic_transactions.csv")
    args = parser.parse_args()
    main(args.data)
