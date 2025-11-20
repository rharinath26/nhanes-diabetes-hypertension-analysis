import os
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

MODEL_DIR = "modeling_outputs"
OUTPUT_DIR = os.path.join("visualizations", "top_features")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_LABELS: Dict[str, str] = {
    "RIDAGEYR": "Age (Years)",
    "RIAGENDR": "Gender",
    "RIDRETH3": "Race Code",
    "INDFMPIR": "Income-Poverty Ratio",
    "BMXBMI": "BMI (kg/m²)",
    "PAD680": "Moderate Activity (min/wk)",
    "PAD800": "Vigorous Activity (min/wk)",
    "DR1TKCAL": "Daily Calories (kcal)",
    "DR1TSODI": "Daily Sodium (mg)",
    "DR1TCARB": "Daily Carbs (g)",
    "DR1TTFAT": "Daily Total Fat (g)",
    "DR1TSFAT": "Daily Saturated Fat (g)",
    "DR1TSUGR": "Daily Sugar (g)",
    "DR1TFIBE": "Daily Fiber (g)",
    "DR1TPROT": "Daily Protein (g)",
    "SMQ020": "Ever Smoked",
    "SMQ040": "Smoking Frequency",
    "ALQ121": "Alcohol Days/Year",
    "ALQ130": "Avg Drinks/Day",
    "LBXGLU": "Fasting Glucose (mg/dL)",
    "LBXGH": "HbA1c (%)",
    "LBXTC": "Total Cholesterol (mg/dL)",
    "LBDHDD": "HDL Cholesterol (mg/dL)",
    "BPXOSY1": "Systolic BP (mmHg)",
    "BPXODI1": "Diastolic BP (mmHg)",
}

TOP_N = 10


def format_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def plot_top_features(df: pd.DataFrame, title: str, filename: str, color: str = "#266dd3"):
    if df.empty:
        return
    df = df.copy()
    df["Feature_Label"] = df["Feature"].map(format_label)
    subset = df.head(TOP_N).iloc[::-1]  # reverse for horizontal plot

    plt.figure(figsize=(9, 6))
    plt.barh(subset["Feature_Label"], subset["Importance"], color=color, alpha=0.85)
    for idx, val in enumerate(subset["Importance"]):
        plt.text(val + subset["Importance"].max() * 0.01, idx, f"{val:.3f}", va="center")
    plt.xlabel("Feature Importance")
    plt.title(title)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_overall():
    overall_files = [
        ("overall_feature_importance_with_age.csv", "Overall (With Age)", "overall_with_age"),
        ("overall_feature_importance_without_age.csv", "Overall (Without Age)", "overall_without_age"),
    ]
    for csv_name, label, prefix in overall_files:
        path = os.path.join(MODEL_DIR, csv_name)
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
        df = pd.read_csv(path)
        for target in df["Target"].unique():
            subset = df[df["Target"] == target].sort_values("Importance", ascending=False)
            title = f"Top {TOP_N} Features - {target} ({label})"
            filename = f"top10_{prefix}_{target.lower()}.png"
            plot_top_features(subset, title, filename, color="#2c82c9")


def plot_race_specific():
    race_files = [
        ("has_diabetes_feature_importance.csv", "Has_Diabetes"),
        ("has_hypertension_feature_importance.csv", "Has_Hypertension"),
    ]
    for csv_name, target in race_files:
        path = os.path.join(MODEL_DIR, csv_name)
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        for race in df["Race_Label"].unique():
            subset = (
                df[df["Race_Label"] == race]
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )
            safe_race = race.lower().replace(" ", "_").replace("/", "_")
            title = f"Top {TOP_N} Features - {target} ({race})"
            filename = f"top10_{target.lower()}_{safe_race}.png"
            plot_top_features(subset, title, filename, color="#16a085")


def main():
    plot_overall()
    plot_race_specific()


if __name__ == "__main__":
    main()
