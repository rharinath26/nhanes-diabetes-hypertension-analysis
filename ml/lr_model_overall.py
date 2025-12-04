import os
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

PARQUET_FOLDER = "parquet"
OUTPUT_DIR = "modeling_outputs"
LR_DIR = os.path.join(OUTPUT_DIR, "logistic_regression")
CURVE_DIR = os.path.join(LR_DIR, "curves")
MODEL_DIR = os.path.join(LR_DIR, "saved_models")

os.makedirs(LR_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MPL_CACHE_DIR = os.path.join(OUTPUT_DIR, ".mplcache")
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

BASE_FEATURES = [
    "RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR", "BMXBMI",
    "PAD680", "PAD800", "DR1TKCAL", "DR1TSODI", "DR1TCARB",
    "DR1TTFAT", "DR1TSFAT", "DR1TSUGR", "DR1TFIBE", "DR1TPROT",
    "SMQ020", "SMQ040", "ALQ121", "ALQ130", "LBXGH", "LBXTC",
    "LBDHDD", "BPXOSY1", "BPXODI1",
]

def get_features(include_age: bool) -> List[str]:
    features = BASE_FEATURES.copy()
    if not include_age and "RIDAGEYR" in features:
        features.remove("RIDAGEYR")
    return features

def load_dataframe() -> pd.DataFrame:
    print("Loading NHANES datasets")
    paths = ["parquet", "ml/parquet", "../eda/parquet", "eda/parquet"]
    parquet_path = None
    for p in paths:
        if os.path.exists(p) and os.path.isdir(p):
            parquet_path = p
            break
    
    if not parquet_path:
        raise FileNotFoundError("Could not find parquet directory")

    demo = pd.read_parquet(os.path.join(parquet_path, "DEMO_L.parquet"))
    diq = pd.read_parquet(os.path.join(parquet_path, "DIQ_L.parquet"))
    bpq = pd.read_parquet(os.path.join(parquet_path, "BPQ_L.parquet"))
    bpxo = pd.read_parquet(os.path.join(parquet_path, "BPXO_L.parquet"))
    bmx = pd.read_parquet(os.path.join(parquet_path, "BMX_L.parquet"))
    dr1tot = pd.read_parquet(os.path.join(parquet_path, "DR1TOT_L.parquet"))
    paq = pd.read_parquet(os.path.join(parquet_path, "PAQ_L.parquet"))
    smq = pd.read_parquet(os.path.join(parquet_path, "SMQ_L.parquet"))
    alq = pd.read_parquet(os.path.join(parquet_path, "ALQ_L.parquet"))
    ghb = pd.read_parquet(os.path.join(parquet_path, "GHB_L.parquet"))
    tchol = pd.read_parquet(os.path.join(parquet_path, "TCHOL_L.parquet"))
    hdl = pd.read_parquet(os.path.join(parquet_path, "HDL_L.parquet"))

    df = demo.copy()
    df = df.merge(diq[["SEQN", "DIQ010"]], on="SEQN", how="left")
    df = df.merge(bpq[["SEQN", "BPQ020"]], on="SEQN", how="left")
    df = df.merge(bpxo[["SEQN", "BPXOSY1", "BPXODI1"]], on="SEQN", how="left")
    df = df.merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="left")
    df = df.merge(dr1tot[["SEQN", "DR1TKCAL", "DR1TSODI", "DR1TCARB", "DR1TTFAT",
                          "DR1TSFAT", "DR1TSUGR", "DR1TFIBE", "DR1TPROT"]], on="SEQN", how="left")
    df = df.merge(paq[["SEQN", "PAD680", "PAD800"]], on="SEQN", how="left")
    df = df.merge(smq[["SEQN", "SMQ020", "SMQ040"]], on="SEQN", how="left")
    df = df.merge(alq[["SEQN", "ALQ121", "ALQ130"]], on="SEQN", how="left")
    df = df.merge(ghb[["SEQN", "LBXGH"]], on="SEQN", how="left")
    df = df.merge(tchol[["SEQN", "LBXTC"]], on="SEQN", how="left")
    df = df.merge(hdl[["SEQN", "LBDHDD"]], on="SEQN", how="left")

    df.loc[df["SMQ020"].isin([7, 9]), "SMQ020"] = np.nan
    df.loc[df["SMQ040"].isin([7, 9]), "SMQ040"] = np.nan
    df.loc[df["ALQ121"] < 0, "ALQ121"] = np.nan
    df.loc[df["ALQ130"] < 0, "ALQ130"] = np.nan
    df.loc[df["PAD680"].isin([7777, 9999]), "PAD680"] = np.nan
    df.loc[df["PAD800"].isin([7777, 9999]), "PAD800"] = np.nan

    df["Has_Diabetes"] = (df["DIQ010"] == 1).astype(float)
    df["Has_Diabetes"] = df["Has_Diabetes"].where(~df["DIQ010"].isin([7, 9]))
    df["Has_Hypertension"] = (df["BPQ020"] == 1).astype(float)
    df["Has_Hypertension"] = df["Has_Hypertension"].where(~df["BPQ020"].isin([7, 9]))

    print(f"Combined dataframe: {len(df):,} rows")
    return df

def save_curves(target: str, label: str, y: np.ndarray, probas: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, probas)
    precision, recall, _ = precision_recall_curve(y, probas)
    roc_auc = auc(fpr, tpr)
    avg_precision = average_precision_score(y, probas)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {target} ({label})")
    plt.legend(loc="lower right")
    roc_path = os.path.join(CURVE_DIR, f"{target.lower()}_{label}_roc.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="steelblue", lw=2, label=f"AP = {avg_precision:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {target} ({label})")
    plt.legend(loc="lower left")
    pr_path = os.path.join(CURVE_DIR, f"{target.lower()}_{label}_pr.png")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()

    return avg_precision

def train_overall_model(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    label: str,
) -> tuple:
    subset = df.dropna(subset=[target])
    X = subset[features]
    y = subset[target].astype(int)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver='lbfgs',
                    random_state=42
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    probas = cross_val_predict(
        pipeline, X, y, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    avg_precision = save_curves(target, label, y, probas)

    pipeline.fit(X, y)
    model = pipeline.named_steps["model"]

    model_path = os.path.join(MODEL_DIR, f"{target.lower()}_{label}_pipeline.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Saved model pipeline: {model_path}")

    coeffs = model.coef_[0]
    feature_df = (
        pd.DataFrame({"Feature": features, "Coefficient": coeffs, "Abs_Coefficient": np.abs(coeffs)})
        .sort_values("Abs_Coefficient", ascending=False)
        .reset_index(drop=True)
    )
    feature_df["Rank"] = feature_df["Abs_Coefficient"].rank(
        method="first", ascending=False
    ).astype(int)
    feature_df["Target"] = target

    metrics = pd.DataFrame(
        [
            {
                "Target": target,
                "Samples": len(subset),
                "ROC_AUC_Mean": roc_auc_scores.mean(),
                "ROC_AUC_STD": roc_auc_scores.std(),
                "PR_AvgPrecision": avg_precision,
                "Age_Included": label == "with_age",
            }
        ]
    )

    print(
        f"{target} ({label}): ROC-AUC {roc_auc_scores.mean():.3f} ± {roc_auc_scores.std():.3f} "
        f"(n={len(subset)}) | AP={avg_precision:.3f}"
    )
    return feature_df, metrics

def run_setting(include_age: bool, df: pd.DataFrame) -> None:
    label = "with_age" if include_age else "without_age"
    features = get_features(include_age)

    all_feat = []
    all_metrics = []

    for target in ["Has_Diabetes", "Has_Hypertension"]:
        feat_df, metric_df = train_overall_model(df, target, features, label)
        all_feat.append(feat_df)
        all_metrics.append(metric_df)

    feature_path = os.path.join(
        LR_DIR, f"lr_overall_feature_importance_{label}.csv"
    )
    metrics_path = os.path.join(LR_DIR, f"lr_overall_cv_metrics_{label}.csv")

    pd.concat(all_feat, ignore_index=True).to_csv(feature_path, index=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(metrics_path, index=False)

    print(f"Saved feature importances to {feature_path}")
    print(f"Saved CV metrics to {metrics_path}")

def main():
    df = load_dataframe()
    run_setting(include_age=True, df=df)
    run_setting(include_age=False, df=df)
    print("\nOverall Logistic Regression modeling complete. Outputs in modeling_outputs/logistic_regression/")

if __name__ == "__main__":
    main()
