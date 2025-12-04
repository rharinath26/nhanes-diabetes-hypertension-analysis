import os
from typing import List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

PARQUET_FOLDER = "parquet"
OUTPUT_DIR = "modeling_outputs"
PERM_DIR = os.path.join(OUTPUT_DIR, "lr_permutation_importance")
MODEL_DIR_OVERALL = os.path.join(OUTPUT_DIR, "logistic_regression", "saved_models")
MODEL_DIR_RACE = os.path.join(OUTPUT_DIR, "logistic_regression_race", "saved_models")

os.makedirs(PERM_DIR, exist_ok=True)

MPL_CACHE_DIR = os.path.join(OUTPUT_DIR, ".mplcache")
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

RUN_MODE = "both"  # "overall", "race", or "both"
MAX_SAMPLES_OVERALL = None
MAX_SAMPLES_RACE = None

BASE_FEATURES = [
    "RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR", "BMXBMI",
    "PAD680", "PAD800", "DR1TKCAL", "DR1TSODI", "DR1TCARB",
    "DR1TTFAT", "DR1TSFAT", "DR1TSUGR", "DR1TFIBE", "DR1TPROT",
    "SMQ020", "SMQ040", "ALQ121", "ALQ130", "LBXGH", "LBXTC",
    "LBDHDD", "BPXOSY1", "BPXODI1",
]

FEATURE_LABELS = {
    "RIDAGEYR": "Age (Years)",
    "RIAGENDR": "Gender",
    "RIDRETH3": "Race Code",
    "INDFMPIR": "Income-Poverty Ratio",
    "BMXBMI": "BMI (kg/m²)",
    "PAD680": "Sedentary Behavior (min/day)",
    "PAD800": "Moderate Activity (min/wk)",
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

def get_features(include_age: bool, include_race: bool) -> List[str]:
    features = BASE_FEATURES.copy()
    if not include_age and "RIDAGEYR" in features:
        features.remove("RIDAGEYR")
    if not include_race and "RIDRETH3" in features:
        features.remove("RIDRETH3")
    return features

def load_dataframe() -> pd.DataFrame:
    print("Loading NHANES data")
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

    # Cleaning
    df.loc[df["SMQ020"].isin([7, 9]), "SMQ020"] = np.nan
    df.loc[df["SMQ040"].isin([7, 9]), "SMQ040"] = np.nan
    df.loc[df["ALQ121"] < 0, "ALQ121"] = np.nan
    df.loc[df["ALQ130"] < 0, "ALQ130"] = np.nan
    df.loc[df["PAD680"].isin([7777, 9999]), "PAD680"] = np.nan
    df.loc[df["PAD800"].isin([7777, 9999]), "PAD800"] = np.nan

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
        7: "Other/Multi-racial",
    }
    df["Race_Label"] = df["RIDRETH3"].map(race_map)

    print(f"Combined dataframe: {len(df):,} rows")
    return df

def stratified_sample(X: pd.DataFrame, y: pd.Series, max_samples: int) -> tuple:
    if max_samples is None or len(X) <= max_samples:
        print(f"  Using all {len(X)} samples")
        return X, y
    print(f"  Sampling {max_samples} of {len(X)} samples (stratified)")
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=max_samples, stratify=y, random_state=42
    )
    print(f"  Sampled class distribution: {y_sample.value_counts().to_dict()}")
    return X_sample, y_sample

def compute_permutation(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    label: str,
    max_samples: int,
    model_dir: str,
    n_repeats: int = 10,
) -> None:
    subset = df.dropna(subset=[target])
    if len(subset) < 100:
        print(f"Skipping {target} ({label}): insufficient samples ({len(subset)})")
        return

    X = subset[features].copy()
    y = subset[target].astype(int)

    model_path = os.path.join(model_dir, f"{target.lower()}_{sanitize_label(label)}_pipeline.pkl")
    if os.path.exists(model_path):
        print(f"  Loading existing model from: {model_path}")
        pipeline = joblib.load(model_path)
        imputer = pipeline.named_steps["imputer"]
        scaler = pipeline.named_steps["scaler"]
        model = pipeline.named_steps["model"]
        X_imputed = pd.DataFrame(
            imputer.transform(X), columns=X.columns, index=X.index
        )
        X_scaled = pd.DataFrame(
            scaler.transform(X_imputed), columns=X.columns, index=X.index
        )
    else:
        print("  No saved model found. Training ad-hoc model for permutation importance")
        imputer = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X), columns=X.columns, index=X.index
        )
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_imputed), columns=X.columns, index=X.index
        )
        model = LogisticRegression(
            class_weight="balanced", max_iter=1000, solver='lbfgs', random_state=42
        )
        model.fit(X_scaled, y)

    X_perm, y_perm = stratified_sample(X_scaled, y, max_samples)
    print(
        f"  Computing permutation importance for {len(X_perm)} samples "
        f"(repeats={n_repeats})"
    )

    X_perm_array = X_perm.to_numpy()

    perm_result = permutation_importance(
        model,
        X_perm_array,
        y_perm,
        n_repeats=n_repeats,
        random_state=42,
        scoring="roc_auc",
        n_jobs=1,
    )

    results_df = pd.DataFrame(
        {
            "Feature": features,
            "Feature_Label": [FEATURE_LABELS.get(f, f) for f in features],
            "Importance_Mean": perm_result.importances_mean,
            "Importance_STD": perm_result.importances_std,
            "Target": target,
            "Label": label,
        }
    ).sort_values("Importance_Mean", ascending=False)
    results_df["Rank"] = results_df["Importance_Mean"].rank(
        method="first", ascending=False
    ).astype(int)
    results_df["Importance_Normalized"] = (
        results_df["Importance_Mean"] / results_df["Importance_Mean"].sum()
    )

    label_safe = sanitize_label(label)
    csv_path = os.path.join(
        PERM_DIR, f"perm_importance_{target.lower()}_{label_safe}.csv"
    )
    results_df.to_csv(csv_path, index=False)
    print(f"  Saved CSV: {csv_path}")

    top_n = 15
    top_features = results_df.head(top_n)
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = np.arange(len(top_features))
    ax.barh(
        y_pos,
        top_features["Importance_Mean"].values,
        xerr=top_features["Importance_STD"].values,
        color="steelblue",
        alpha=0.75,
        edgecolor="black",
        capsize=3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features["Feature_Label"].values, fontsize=10)
    ax.set_xlabel("Permutation Importance (Δ ROC-AUC)")
    ax.set_title(
        f"Permutation Importance - {target.replace('Has_', '')} ({label})\nTop {top_n} Features (Logistic Regression)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    plot_path = os.path.join(
        PERM_DIR, f"perm_importance_{target.lower()}_{label_safe}.png"
    )
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {plot_path}")

def analyze_race(
    df: pd.DataFrame,
    target: str,
    features: List[str],
    max_samples: int,
    min_samples: int = 300,
):
    for race_label, subset in df.groupby("Race_Label"):
        if pd.isna(race_label):
            continue
        subset = subset.dropna(subset=[target])
        if len(subset) < min_samples:
            print(f"Skipping {race_label}: insufficient samples ({len(subset)})")
            continue
        print(f"\nAnalyzing {race_label} ({target})")
        compute_permutation(subset, target, features, race_label, max_samples, MODEL_DIR_RACE)

def main():
    print("Permutation Importance Analysis (Logistic Regression)")

    df = load_dataframe()
    targets = ["Has_Diabetes", "Has_Hypertension"]

    if RUN_MODE in {"overall", "both"}:
        print("\nRunning overall models")
        for include_age in [True, False]:
            label = "with_age" if include_age else "without_age"
            features = get_features(include_age=include_age, include_race=True)
            for target in targets:
                print(f"\n{'-'*40}\nOverall: {target} ({label})")
                compute_permutation(
                    df, target, features, label, max_samples=MAX_SAMPLES_OVERALL, model_dir=MODEL_DIR_OVERALL
                )

    if RUN_MODE in {"race", "both"}:
        print("\nRunning race-specific models")
        features_no_age = get_features(include_age=False, include_race=False)
        for target in targets:
            print(f"\n{'-'*40}\nRace-specific: {target}")
            analyze_race(df, target, features_no_age, max_samples=MAX_SAMPLES_RACE)

    print(f"\nPermutation importance complete. Outputs in {PERM_DIR}")

if __name__ == "__main__":
    main()
