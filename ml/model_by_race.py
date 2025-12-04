import os
from typing import Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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
CURVE_DIR = os.path.join(OUTPUT_DIR, "curves_by_race")
MODEL_DIR = os.path.join(OUTPUT_DIR, "saved_models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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
    print("Loading datasets for modeling")
    demo = pd.read_parquet(os.path.join(PARQUET_FOLDER, "DEMO_L.parquet"))
    diq = pd.read_parquet(os.path.join(PARQUET_FOLDER, "DIQ_L.parquet"))
    bpq = pd.read_parquet(os.path.join(PARQUET_FOLDER, "BPQ_L.parquet"))
    bpxo = pd.read_parquet(os.path.join(PARQUET_FOLDER, "BPXO_L.parquet"))
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
    df = df.merge(bpxo[["SEQN", "BPXOSY1", "BPXODI1"]], on="SEQN", how="left")
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

    df.loc[df["SMQ020"].isin([7, 9]), "SMQ020"] = np.nan
    df.loc[df["SMQ040"].isin([7, 9]), "SMQ040"] = np.nan
    df.loc[df["ALQ121"] < 0, "ALQ121"] = np.nan
    df.loc[df["ALQ130"] < 0, "ALQ130"] = np.nan

    df["Has_Diabetes"] = (df["DIQ010"] == 1).astype(float)
    df["Has_Diabetes"] = df["Has_Diabetes"].where(~df["DIQ010"].isin([7, 9]))
    df["Has_Hypertension"] = (df["BPQ020"] == 1).astype(float)
    df["Has_Hypertension"] = df["Has_Hypertension"].where(~df["BPQ020"].isin([7, 9]))

    df["Race_Label"] = df["RIDRETH3"].map(RACE_LABELS)
    print(f"Modeling dataframe: {len(df):,} rows, {len(df.columns):,} columns")
    return df


def train_group_models(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    min_samples: int = 300,
) -> Dict[str, pd.DataFrame]:
    results = []
    metrics = []

    for race_label, group_df in df.groupby("Race_Label"):
        if pd.isna(race_label):
            continue
        subset = group_df.dropna(subset=[target_col])
        if len(subset) < min_samples:
            print(f"Skipping {race_label} for {target_col}: insufficient samples ({len(subset)})")
            continue

        X = subset[feature_cols]
        y = subset[target_col].astype(int)

        pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        random_state=42,
                        class_weight="balanced",
                        n_jobs=1,
                    ),
                ),
            ]
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        roc_auc_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1)
        probas = cross_val_predict(
            pipeline, X, y, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        avg_precision = average_precision_score(y, probas)

        fpr, tpr, _ = roc_curve(y, probas)
        precision, recall, _ = precision_recall_curve(y, probas)
        roc_auc = auc(fpr, tpr)

        race_safe = sanitize_label(race_label)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {target_col} ({race_label})")
        plt.legend(loc="lower right")
        roc_path = os.path.join(
            CURVE_DIR, f"{target_col.lower()}_{race_safe}_roc.png"
        )
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color="steelblue", lw=2, label=f"AP = {avg_precision:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve - {target_col} ({race_label})")
        plt.legend(loc="lower left")
        pr_path = os.path.join(
            CURVE_DIR, f"{target_col.lower()}_{race_safe}_pr.png"
        )
        plt.tight_layout()
        plt.savefig(pr_path, dpi=300, bbox_inches="tight")
        plt.close()

        pipeline.fit(X, y)
        model = pipeline.named_steps["model"]
        feature_importances = model.feature_importances_

        model_path = os.path.join(
            MODEL_DIR, f"{target_col.lower()}_{race_safe}_pipeline.pkl"
        )
        joblib.dump(pipeline, model_path)

        for feat, importance in zip(feature_cols, feature_importances):
            results.append(
                {
                    "Target": target_col,
                    "Race_Label": race_label,
                    "Feature": feat,
                    "Importance": importance,
                }
            )

        metrics.append(
            {
                "Target": target_col,
                "Race_Label": race_label,
                "Samples": len(subset),
                "ROC_AUC_Mean": roc_auc_scores.mean(),
                "ROC_AUC_STD": roc_auc_scores.std(),
                "PR_AvgPrecision": avg_precision,
            }
        )

        print(
            f"{target_col} | {race_label}: ROC-AUC {roc_auc_scores.mean():.3f} ± "
            f"{roc_auc_scores.std():.3f} (n={len(subset)}) | AP={avg_precision:.3f}"
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
    df = df[df["Race_Label"].notna()].copy()

    available_features = [col for col in FEATURE_COLUMNS if col in df.columns]
    missing = sorted(set(FEATURE_COLUMNS) - set(available_features))
    if missing:
        print("Warning: missing features skipped ->", missing)

    outputs = {}
    for target in ["Has_Diabetes", "Has_Hypertension"]:
        outputs[target] = train_group_models(df, target, available_features)

    for target, data in outputs.items():
        feature_path = os.path.join(
            OUTPUT_DIR, f"{target.lower()}_feature_importance.csv"
        )
        metrics_path = os.path.join(OUTPUT_DIR, f"{target.lower()}_cv_metrics.csv")
        data["feature_importance"].to_csv(feature_path, index=False)
        data["cv_metrics"].to_csv(metrics_path, index=False)
        print(f"Saved {feature_path}")
        print(f"Saved {metrics_path}")

    print("\nRace-stratified Random Forest modeling complete.")


if __name__ == "__main__":
    main()
