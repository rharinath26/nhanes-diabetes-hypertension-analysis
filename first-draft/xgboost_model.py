import os
from typing import List, Dict
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    average_precision_score
)

PARQUET_FOLDER = "parquet"
OUTPUT_DIR = "modeling_outputs"
XGBOOST_DIR = os.path.join(OUTPUT_DIR, "xgboost")
CURVE_DIR = os.path.join(XGBOOST_DIR, "curves")
os.makedirs(XGBOOST_DIR, exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)

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
    "BPXODI1"
]

FEATURE_LABELS = {
    "RIDAGEYR": "Age (Years)",
    "RIAGENDR": "Gender",
    "RIDRETH3": "Race Code",
    "INDFMPIR": "Income-Poverty Ratio",
    "BMXBMI": "BMI (kg/m²)",
    "PAD680": "Moderate Activity (min/wk)",
    "PAD800": "Vigorous Activity (min/wk)",
    "DR1TKCAL": "Daily Calories (kcal)",
    "DR1TSODI": "Daily Sodium (mg)",
    "DR1TCARB": "Daily Carbohydrates (g)",
    "DR1TTFAT": "Daily Total Fat (g)",
    "DR1TSFAT": "Daily Saturated Fat (g)",
    "DR1TSUGR": "Daily Sugar (g)",
    "DR1TFIBE": "Daily Fiber (g)",
    "DR1TPROT": "Daily Protein (g)",
    "SMQ020": "Ever Smoked",
    "SMQ040": "Smoking Frequency",
    "ALQ121": "Alcohol Days/Year",
    "ALQ130": "Avg Drinks/Day",
    "LBXGH": "HbA1c (%)",
    "LBXTC": "Total Cholesterol (mg/dL)",
    "LBDHDD": "HDL Cholesterol (mg/dL)",
    "BPXOSY1": "Systolic BP (mmHg)",
    "BPXODI1": "Diastolic BP (mmHg)",
}


def sanitize_label(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )


def get_features(include_age: bool) -> List[str]:
    features = [f for f in BASE_FEATURES if f != "RIDAGEYR"] if not include_age else BASE_FEATURES.copy()
    return features


def load_dataframe() -> pd.DataFrame:
    print("Loading NHANES data...")
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
    df = df.merge(dr1tot[["SEQN", "DR1TKCAL", "DR1TSODI", "DR1TCARB", "DR1TTFAT",
                          "DR1TSFAT", "DR1TSUGR", "DR1TFIBE", "DR1TPROT"]], on="SEQN", how="left")
    df = df.merge(paq[["SEQN", "PAD680", "PAD800"]], on="SEQN", how="left")
    df = df.merge(smq[["SEQN", "SMQ020", "SMQ040"]], on="SEQN", how="left")
    df = df.merge(alq[["SEQN", "ALQ121", "ALQ130"]], on="SEQN", how="left")
    df = df.merge(ghb[["SEQN", "LBXGH"]], on="SEQN", how="left")
    df = df.merge(tchol[["SEQN", "LBXTC"]], on="SEQN", how="left")
    df = df.merge(hdl[["SEQN", "LBDHDD"]], on="SEQN", how="left")

    df["Has_Diabetes"] = (df["DIQ010"] == 1).astype(float)
    df["Has_Hypertension"] = (df["BPQ020"] == 1).astype(float)
    df["Has_Diabetes"] = df["Has_Diabetes"].where(~df["DIQ010"].isin([7, 9]))
    df["Has_Hypertension"] = df["Has_Hypertension"].where(~df["BPQ020"].isin([7, 9]))

    race_map = {
        1: "Mexican American",
        2: "Other Hispanic",
        3: "Non-Hispanic White",
        4: "Non-Hispanic Black",
        6: "Non-Hispanic Asian",
        7: "Other/Multi-racial"
    }
    df["Race_Label"] = df["RIDRETH3"].map(race_map)

    print(f"Combined dataframe: {len(df):,} rows")
    return df


def train_overall_model(df: pd.DataFrame, target: str, features: List[str], label: str) -> tuple:

    subset = df.dropna(subset=[target])
    X = subset[features]
    y = subset[target].astype(int)

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_jobs=1,
            eval_metric="logloss"
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    roc_auc_scores = cross_val_score(
        pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1
    )
    probas = cross_val_predict(
        pipeline, X, y, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    fpr, tpr, _ = roc_curve(y, probas)
    precision, recall, _ = precision_recall_curve(y, probas)
    roc_auc = auc(fpr, tpr)
    avg_precision = average_precision_score(y, probas)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {target} ({label})')
    plt.legend(loc="lower right")
    roc_path = os.path.join(CURVE_DIR, f"{target.lower()}_{label}_roc.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {target} ({label})')
    plt.legend(loc="lower left")
    pr_path = os.path.join(CURVE_DIR, f"{target.lower()}_{label}_pr.png")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300, bbox_inches="tight")
    plt.close()

    pipeline.fit(X, y)
    model = pipeline.named_steps["model"]

    feature_df = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False)
    feature_df["Importance_Normalized"] = feature_df["Importance"] / feature_df["Importance"].sum()
    feature_df["Rank"] = feature_df["Importance"].rank(method="first", ascending=False).astype(int)
    feature_df["Target"] = target
    feature_df["Model"] = "XGBoost"

    metrics = pd.DataFrame([{
        "Target": target,
        "Model": "XGBoost",
        "Samples": len(subset),
        "ROC_AUC_Mean": roc_auc_scores.mean(),
        "ROC_AUC_STD": roc_auc_scores.std(),
        "PR_AvgPrecision": avg_precision,
        "Age_Included": "with_age" in label
    }])

    print(f"{target} ({label}): ROC-AUC {roc_auc_scores.mean():.3f} ± {roc_auc_scores.std():.3f} ({len(subset)} samples)")
    return feature_df, metrics


def train_group_models(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    min_samples: int = 300
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

        # Calculate class weights
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                n_jobs=1,
                eval_metric="logloss"
            ))
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        roc_auc_scores = cross_val_score(
            pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=1
        )
        probas = cross_val_predict(
            pipeline, X, y, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        fpr, tpr, _ = roc_curve(y, probas)
        precision, recall, _ = precision_recall_curve(y, probas)
        roc_auc = auc(fpr, tpr)
        avg_precision = average_precision_score(y, probas)

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {target_col} ({race_label})')
        plt.legend(loc="lower right")
        race_safe = sanitize_label(race_label)
        roc_path = os.path.join(CURVE_DIR, f"{target_col.lower()}_{race_safe}_roc.png")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {target_col} ({race_label})')
        plt.legend(loc="lower left")
        pr_path = os.path.join(CURVE_DIR, f"{target_col.lower()}_{race_safe}_pr.png")
        plt.tight_layout()
        plt.savefig(pr_path, dpi=300, bbox_inches="tight")
        plt.close()

        pipeline.fit(X, y)
        model = pipeline.named_steps["model"]
        feature_importances = model.feature_importances_

        for feat, importance in zip(feature_cols, feature_importances):
            results.append({
                "Feature": feat,
                "Importance": importance,
                "Target": target_col,
                "Group": race_label,
                "Model": "XGBoost"
            })

        metrics.append({
            "Target": target_col,
            "Group": race_label,
            "Model": "XGBoost",
            "Samples": len(subset),
            "ROC_AUC_Mean": roc_auc_scores.mean(),
            "ROC_AUC_STD": roc_auc_scores.std(),
            "PR_AvgPrecision": avg_precision
        })

        print(f"  {race_label}: ROC-AUC {roc_auc_scores.mean():.3f} ± {roc_auc_scores.std():.3f}")

    feature_df = pd.DataFrame(results)
    if not feature_df.empty:
        feature_df["Importance_Normalized"] = feature_df.groupby(["Target", "Group"])["Importance"].transform(
            lambda x: x / x.sum()
        )
        feature_df["Rank"] = feature_df.groupby(["Target", "Group"])["Importance"].rank(
            method="first", ascending=False
        ).astype(int)

    metrics_df = pd.DataFrame(metrics)
    return {"feature_importance": feature_df, "cv_metrics": metrics_df}


def main():
    print("XGBoost Model Training")

    df = load_dataframe()
    targets = ["Has_Diabetes", "Has_Hypertension"]

    all_feature_dfs = []
    all_metrics_dfs = []

    # Overall models (with and without age)
    for include_age in [True, False]:
        age_label = "with_age" if include_age else "without_age"
        features = get_features(include_age)

        for target in targets:
            print(f"\nTraining overall model: {target} ({age_label})")
            feature_df, metrics_df = train_overall_model(df, target, features, age_label)
            if feature_df is not None:
                all_feature_dfs.append(feature_df)
                all_metrics_dfs.append(metrics_df)

    if all_feature_dfs:
        overall_features = pd.concat(all_feature_dfs, ignore_index=True)
        overall_features.to_csv(
            os.path.join(XGBOOST_DIR, "xgboost_overall_feature_importance.csv"),
            index=False
        )
        print(f"\nSaved overall feature importance to: {XGBOOST_DIR}/xgboost_overall_feature_importance.csv")

    if all_metrics_dfs:
        overall_metrics = pd.concat(all_metrics_dfs, ignore_index=True)
        overall_metrics.to_csv(
            os.path.join(XGBOOST_DIR, "xgboost_overall_cv_metrics.csv"),
            index=False
        )
        print(f"Saved overall metrics to: {XGBOOST_DIR}/xgboost_overall_cv_metrics.csv")

    features_no_age = get_features(include_age=False)
    for target in targets:
        print(f"\nTraining race-specific models: {target}")
        results = train_group_models(df, target, features_no_age)
        if results:
            if not results["feature_importance"].empty:
                results["feature_importance"].to_csv(
                    os.path.join(XGBOOST_DIR, f"xgboost_{target.lower()}_feature_importance.csv"),
                    index=False
                )
            if not results["cv_metrics"].empty:
                results["cv_metrics"].to_csv(
                    os.path.join(XGBOOST_DIR, f"xgboost_{target.lower()}_cv_metrics.csv"),
                    index=False
                )

    print("XGBoost Modeling Complete.")
    print(f"Outputs saved to: {XGBOOST_DIR}")


if __name__ == "__main__":
    main()
