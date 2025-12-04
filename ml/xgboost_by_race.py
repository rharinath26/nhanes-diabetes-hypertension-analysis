import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

PARQUET_FOLDER = "parquet"
OUTPUT_DIR = "modeling_outputs/xgboost/xgboost_by_race"
CURVE_DIR = os.path.join(OUTPUT_DIR, "curves")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)

MPL_CACHE_DIR = os.path.join(OUTPUT_DIR, ".mplcache")
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

RACE_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    6: "Non-Hispanic Asian",
    7: "Other/Multi-racial",
}

FEATURE_COLUMNS = [
    "RIAGENDR",
    "INDFMPIR",
    "BMXBMI",
    "PAD680",
    "PAD800",
    "DR1TKCAL",
    "DR1TSODI",
    "DR1TCARB",
    "DR1TTFAT",
    "DR1TSFAT",
    "DR1TSUGR",
    "DR1TFIBE",
    "DR1TPROT",
    "SMQ020",
    "SMQ040",
    "ALQ121",
    "ALQ130",
    "LBXGH",
    "LBXTC",
    "LBDHDD",
    "BPXOSY1",
    "BPXODI1",
]


def sanitize_label(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def load_dataframe() -> pd.DataFrame:
    demo = pd.read_parquet(os.path.join(PARQUET_FOLDER, "DEMO_L.parquet"))
    diq = pd.read_parquet(os.path.join(PARQUET_FOLDER, "DIQ_L.parquet"))
    bpq = pd.read_parquet(os.path.join(PARQUET_FOLDER, "BPQ_L.parquet"))
    bmx = pd.read_parquet(os.path.join(PARQUET_FOLDER, "BMX_L.parquet"))
    dr1tot = pd.read_parquet(os.path.join(PARQUET_FOLDER, "DR1TOT_L.parquet"))
    paq = pd.read_parquet(os.path.join(PARQUET_FOLDER, "PAQ_L.parquet"))
    smq = pd.read_parquet(os.path.join(PARQUET_FOLDER, "SMQ_L.parquet"))
    alq = pd.read_parquet(os.path.join(PARQUET_FOLDER, "ALQ_L.parquet"))
    ghb = pd.read_parquet(os.path.join(PARQUET_FOLDER, "GHB_L.parquet"))
    tchol = pd.read_parquet(os.path.join(PARQUET_FOLDER, "TCHOL_L.parquet"))
    hdl = pd.read_parquet(os.path.join(PARQUET_FOLDER, "HDL_L.parquet"))

    df = demo.copy()
    df = df.merge(diq[["SEQN", "DIQ010"]], on="SEQN", how="left")
    df = df.merge(bpq[["SEQN", "BPQ020"]], on="SEQN", how="left")
    df = df.merge(bmx[["SEQN", "BMXBMI"]], on="SEQN", how="left")
    df = df.merge(
        dr1tot[
            [
                "SEQN",
                "DR1TKCAL",
                "DR1TSODI",
                "DR1TCARB",
                "DR1TTFAT",
                "DR1TSFAT",
                "DR1TSUGR",
                "DR1TFIBE",
                "DR1TPROT",
            ]
        ],
        on="SEQN",
        how="left",
    )
    df = df.merge(paq[["SEQN", "PAD680", "PAD800"]], on="SEQN", how="left")
    df = df.merge(smq[["SEQN", "SMQ020", "SMQ040"]], on="SEQN", how="left")
    df = df.merge(alq[["SEQN", "ALQ121", "ALQ130"]], on="SEQN", how="left")
    df = df.merge(ghb[["SEQN", "LBXGH"]], on="SEQN", how="left")
    df = df.merge(tchol[["SEQN", "LBXTC"]], on="SEQN", how="left")
    df = df.merge(hdl[["SEQN", "LBDHDD"]], on="SEQN", how="left")

    df["Has_Diabetes"] = (df["DIQ010"] == 1).astype(float)
    df["Has_Diabetes"] = df["Has_Diabetes"].where(~df["DIQ010"].isin([7, 9]))
    df["Has_Hypertension"] = (df["BPQ020"] == 1).astype(float)
    df["Has_Hypertension"] = df["Has_Hypertension"].where(~df["BPQ020"].isin([7, 9]))

    df["Race_Label"] = df["RIDRETH3"].map(RACE_LABELS)
    return df[df["Race_Label"].notna()].copy()


def plot_curves(y: np.ndarray, probas: np.ndarray, target: str, race_label: str):
    fpr, tpr, _ = roc_curve(y, probas)
    precision, recall, _ = precision_recall_curve(y, probas)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y, probas)

    race_safe = sanitize_label(race_label)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC - {target} ({race_label})")
    plt.legend()
    path = os.path.join(CURVE_DIR, f"{target.lower()}_{race_safe}_roc.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, lw=2, label=f"AP={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR - {target} ({race_label})")
    plt.legend()
    path = os.path.join(CURVE_DIR, f"{target.lower()}_{race_safe}_pr.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

    return roc_auc, pr_auc


def train_group_models(
    df: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    min_samples: int = 300,
) -> Dict[str, pd.DataFrame]:

    results = []
    metrics = []

    for race_label, group in df.groupby("Race_Label"):
        subset = group.dropna(subset=[target])
        if len(subset) < min_samples:
            print(f"Skipping {race_label} for {target}: only {len(subset)} samples.")
            continue

        X = subset[feature_cols]
        y = subset[target].astype(int)

        n_pos = y.sum()
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=500,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        scale_pos_weight=scale_pos_weight,
                        n_jobs=1,
                        eval_metric="logloss",
                    ),
                ),
            ]
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        roc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        probas = cross_val_predict(
            pipeline, X, y, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        roc_auc, pr_auc = plot_curves(y, probas, target, race_label)

        pipeline.fit(X, y)
        model = pipeline.named_steps["model"]
        feature_importances = model.feature_importances_

        for feat, importance in zip(feature_cols, feature_importances):
            results.append(
                {
                    "Target": target,
                    "Race_Label": race_label,
                    "Feature": feat,
                    "Importance": importance,
                }
            )

        metrics.append(
            {
                "Target": target,
                "Race_Label": race_label,
                "Samples": len(subset),
                "ROC_AUC_CV_Mean": roc_scores.mean(),
                "ROC_AUC_CV_STD": roc_scores.std(),
                "ROC_AUC_Plot": roc_auc,
                "PR_AUC_Plot": pr_auc,
            }
        )

        print(
            f"{target} | {race_label}: ROC-AUC {roc_scores.mean():.3f} ± {roc_scores.std():.3f} "
            f"(n={len(subset)})"
        )

    feature_df = pd.DataFrame(results)
    if not feature_df.empty:
        feature_df["Importance_Normalized"] = feature_df.groupby(
            ["Target", "Race_Label"]
        )["Importance"].transform(lambda x: x / x.sum())
        feature_df["Rank"] = feature_df.groupby(["Target", "Race_Label"])["Importance"].rank(
            method="first", ascending=False
        ).astype(int)

    metrics_df = pd.DataFrame(metrics)
    return {"feature_importance": feature_df, "cv_metrics": metrics_df}


def main():

    df = load_dataframe()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    missing = sorted(set(FEATURE_COLUMNS) - set(feature_cols))
    if missing:
        print("Warning: missing features skipped:", missing)

    outputs = {}
    for target in ["Has_Diabetes", "Has_Hypertension"]:
        outputs[target] = train_group_models(df, target, feature_cols)

    for target, data in outputs.items():
        if not data:
            continue
        feature_path = os.path.join(
            OUTPUT_DIR, f"xgb_{target.lower()}_feature_importance.csv"
        )
        metrics_path = os.path.join(
            OUTPUT_DIR, f"xgb_{target.lower()}_cv_metrics.csv"
        )
        data["feature_importance"].to_csv(feature_path, index=False)
        data["cv_metrics"].to_csv(metrics_path, index=False)
        print(f"Saved {feature_path}")
        print(f"Saved {metrics_path}")

    print("\nXGBoost race-specific modeling complete.")


if __name__ == "__main__":
    main()

