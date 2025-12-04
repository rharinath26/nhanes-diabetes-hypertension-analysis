import os
from typing import List

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
CURVE_DIR = os.path.join(OUTPUT_DIR, "curves_overall")
MODEL_DIR = os.path.join(OUTPUT_DIR, "saved_models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MPL_CACHE_DIR = os.path.join(OUTPUT_DIR, ".mplcache")
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

BASE_FEATURES = [
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH3",
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


def get_features(include_age: bool) -> List[str]:
    features = BASE_FEATURES.copy()
    if not include_age and "RIDAGEYR" in features:
        features.remove("RIDAGEYR")
    return features


def load_dataframe() -> pd.DataFrame:
    print("Loading NHANES datasets")
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = df.dropna(subset=[target])
    X = subset[features]
    y = subset[target].astype(int)

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
    avg_precision = save_curves(target, label, y, probas)

    pipeline.fit(X, y)
    model = pipeline.named_steps["model"]

    model_path = os.path.join(MODEL_DIR, f"{target.lower()}_{label}_pipeline.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Saved model pipeline: {model_path}")

    feature_df = (
        pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )
    feature_df["Importance_Normalized"] = (
        feature_df["Importance"] / feature_df["Importance"].sum()
    )
    feature_df["Rank"] = feature_df["Importance"].rank(
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
        OUTPUT_DIR, f"overall_feature_importance_{label}.csv"
    )
    metrics_path = os.path.join(OUTPUT_DIR, f"overall_cv_metrics_{label}.csv")

    pd.concat(all_feat, ignore_index=True).to_csv(feature_path, index=False)
    pd.concat(all_metrics, ignore_index=True).to_csv(metrics_path, index=False)

    print(f"Saved feature importances to {feature_path}")
    print(f"Saved CV metrics to {metrics_path}")


def main():
    df = load_dataframe()
    run_setting(include_age=True, df=df)
    run_setting(include_age=False, df=df)
    print("\nOverall Random Forest modeling complete. Outputs in modeling_outputs/")


if __name__ == "__main__":
    main()
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


PARQUET_FOLDER = "parquet"
OUTPUT_DIR = "modeling_outputs"
INCLUDE_AGE = False

BASE_FEATURES = [
    "RIDAGEYR",
    "RIAGENDR",
    "RIDRETH3",
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
    "BPXODI1"
]


def get_features() -> List[str]:
    features = BASE_FEATURES.copy()
    if not INCLUDE_AGE and "RIDAGEYR" in features:
        features.remove("RIDAGEYR")
    return features


def load_dataframe() -> pd.DataFrame:
    print("Loading datasets")
    demo = pd.read_parquet(os.path.join(PARQUET_FOLDER, "DEMO_L.parquet"))
    diq = pd.read_parquet(os.path.join(PARQUET_FOLDER, "DIQ_L.parquet"))
    bpq = pd.read_parquet(os.path.join(PARQUET_FOLDER, "BPQ_L.parquet"))
    bpxo = pd.read_parquet(os.path.join(PARQUET_FOLDER, "BPXO_L.parquet"))
    bmx = pd.read_parquet(os.path.join(PARQUET_FOLDER, "BMX_L.parquet"))
    dr1tot = pd.read_parquet(os.path.join(PARQUET_FOLDER, "DR1TOT_L.parquet"))
    paq = pd.read_parquet(os.path.join(PARQUET_FOLDER, "PAQ_L.parquet"))
    smq = pd.read_parquet(os.path.join(PARQUET_FOLDER, "SMQ_L.parquet"))
    alq = pd.read_parquet(os.path.join(PARQUET_FOLDER, "ALQ_L.parquet"))
    glu = pd.read_parquet(os.path.join(PARQUET_FOLDER, "GLU_L.parquet"))
    ghb = pd.read_parquet(os.path.join(PARQUET_FOLDER, "GHB_L.parquet"))
    tchol = pd.read_parquet(os.path.join(PARQUET_FOLDER, "TCHOL_L.parquet"))
    hdl = pd.read_parquet(os.path.join(PARQUET_FOLDER, "HDL_L.parquet"))

    df = demo.copy()
    df = df.merge(diq[["SEQN", "DIQ010"]], on="SEQN", how="left")
    df = df.merge(bpq[["SEQN", "BPQ020"]], on="SEQN", how="left")
    df = df.merge(bpxo[["SEQN", "BPXOSY1", "BPXODI1"]], on="SEQN", how="left")
    df = df.merge(bmx[["SEQN", "BMXBMI", "BMXWAIST"]], on="SEQN", how="left")
    df = df.merge(dr1tot[["SEQN", "DR1TKCAL", "DR1TSODI", "DR1TCARB", "DR1TTFAT",
                          "DR1TSFAT", "DR1TSUGR", "DR1TFIBE", "DR1TPROT"]],
                  on="SEQN", how="left")
    df = df.merge(paq[["SEQN", "PAD680", "PAD800"]], on="SEQN", how="left")
    df = df.merge(smq[["SEQN", "SMQ020", "SMQ040"]], on="SEQN", how="left")
    df = df.merge(alq[["SEQN", "ALQ121", "ALQ130"]], on="SEQN", how="left")
    df = df.merge(glu[["SEQN", "LBXGLU"]], on="SEQN", how="left")
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

    print(f"Combined dataframe: {len(df):,} rows")
    return df


def train_overall_model(df: pd.DataFrame, target: str, features: List[str]) -> pd.DataFrame:
    subset = df.dropna(subset=[target])
    X = subset[features]
    y = subset[target].astype(int)

    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            n_jobs=1
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_auc_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1
    )

    pipeline.fit(X, y)
    model = pipeline.named_steps["model"]

    feature_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    feature_df["Importance_Normalized"] = feature_df["Importance"] / feature_df["Importance"].sum()
    feature_df["Rank"] = feature_df["Importance"].rank(method="first", ascending=False).astype(int)
    feature_df["Target"] = target

    metrics = pd.DataFrame([{
        "Target": target,
        "Samples": len(subset),
        "ROC_AUC_Mean": roc_auc_scores.mean(),
        "ROC_AUC_STD": roc_auc_scores.std()
    }])

    print(f"{target}: ROC-AUC {roc_auc_scores.mean():.3f} ± {roc_auc_scores.std():.3f} ({len(subset)} samples)")
    return feature_df, metrics


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    features = get_features()

    df = load_dataframe()

    all_results = []
    all_metrics = []

    for target in ["Has_Diabetes", "Has_Hypertension"]:
        feats, mets = train_overall_model(df, target, features)
        all_results.append(feats)
        all_metrics.append(mets)

    feature_df = pd.concat(all_results, ignore_index=True)
    metrics_df = pd.concat(all_metrics, ignore_index=True)

    feature_path = os.path.join(OUTPUT_DIR, "overall_feature_importance.csv")
    metrics_path = os.path.join(OUTPUT_DIR, "overall_cv_metrics.csv")

    feature_df.to_csv(feature_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Saved {feature_path}")
    print(f"Saved {metrics_path}")


if __name__ == "__main__":
    main()
