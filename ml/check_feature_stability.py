import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from collections import defaultdict

PARQUET_FOLDER = "parquet"
OUTPUT_DIR = "modeling_outputs"
STABILITY_DIR = os.path.join(OUTPUT_DIR, "stability")
os.makedirs(STABILITY_DIR, exist_ok=True)

BASE_FEATURES = [
    "RIDAGEYR", "RIAGENDR", "RIDRETH3", "INDFMPIR", "BMXBMI",
    "PAD680", "PAD800", "DR1TKCAL", "DR1TSODI", "DR1TCARB",
    "DR1TTFAT", "DR1TSFAT", "DR1TSUGR", "DR1TFIBE", "DR1TPROT",
    "SMQ020", "SMQ040", "ALQ121", "ALQ130", "LBXGH", "LBXTC",
    "LBDHDD", "BPXOSY1", "BPXODI1"
]

FEATURE_LABELS = {
    "RIDAGEYR": "Age",
    "RIAGENDR": "Gender",
    "RIDRETH3": "Race",
    "INDFMPIR": "Income_Ratio",
    "BMXBMI": "BMI",
    "PAD680": "Sedentary_Behavior",
    "PAD800": "Mod_Activity",
    "DR1TKCAL": "Calories",
    "DR1TSODI": "Sodium",
    "DR1TCARB": "Carbs",
    "DR1TTFAT": "Fat",
    "DR1TSFAT": "Sat_Fat",
    "DR1TSUGR": "Sugar",
    "DR1TFIBE": "Fiber",
    "DR1TPROT": "Protein",
    "SMQ020": "Ever_Smoked",
    "SMQ040": "Smoke_Freq",
    "ALQ121": "Alc_Freq",
    "ALQ130": "Alc_Amt",
    "LBXGH": "HbA1c",
    "LBXTC": "Cholesterol",
    "LBDHDD": "HDL",
    "BPXOSY1": "Sys_BP",
    "BPXODI1": "Dia_BP",
}

def load_dataframe():
    print("Loading data")
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
    
    return df

def check_stability(df, target, features, model_type="lr", n_splits=10):
    print(f"\nChecking stability for {target} using {model_type.upper()} (n_splits={n_splits})")
    
    results = []
    
    for race_label, group_df in df.groupby("Race_Label"):
        if pd.isna(race_label): continue
        
        subset = group_df.dropna(subset=[target])
        if len(subset) < 200:
            print(f"Skipping {race_label}: too few samples ({len(subset)})")
            continue
            
        X = subset[features]
        y = subset[target].astype(int)
        
        if y.sum() < 10:
            print(f"Skipping {race_label}: too few positive cases ({y.sum()})")
            continue

        feature_ranks = defaultdict(list)
        
        sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
        
        for split_idx, (train_index, test_index) in enumerate(sss.split(X, y)):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            
            if model_type == "lr":
                pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42 + split_idx
                    ))
                ])
                pipeline.fit(X_train, y_train)
                model = pipeline.named_steps["model"]
                importances = np.abs(model.coef_[0])
            elif model_type == "rf":
                pipeline = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", RandomForestClassifier(
                        n_estimators=100,
                        max_depth=None,
                        class_weight="balanced",
                        n_jobs=1,
                        random_state=42 + split_idx
                    ))
                ])
                pipeline.fit(X_train, y_train)
                model = pipeline.named_steps["model"]
                importances = model.feature_importances_
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            
            indices = np.argsort(importances)[::-1]
            ranks = np.empty_like(indices)
            ranks[indices] = np.arange(1, len(importances) + 1)
            
            for i, feat in enumerate(features):
                feature_ranks[feat].append(ranks[i])
        
        for feat, ranks in feature_ranks.items():
            readable_feat = FEATURE_LABELS.get(feat, feat)
            results.append({
                "Race": race_label,
                "Target": target,
                "Feature": readable_feat,
                "Model": model_type.upper(),
                "Mean_Rank": np.mean(ranks),
                "Std_Rank": np.std(ranks),
                "Min_Rank": np.min(ranks),
                "Max_Rank": np.max(ranks),
                "Top_3_Freq": sum(r <= 3 for r in ranks) / n_splits
            })
            
    return pd.DataFrame(results)

def main():
    df = load_dataframe()
    features_no_age = [f for f in BASE_FEATURES if f != "RIDAGEYR"]

    print("Running Logistic Regression Stability Analysis")
    lr_stab_diabetes = check_stability(df, "Has_Diabetes", features_no_age, model_type="lr")
    lr_stab_hyper = check_stability(df, "Has_Hypertension", features_no_age, model_type="lr")

    print("Running Random Forest Stability Analysis")
    rf_stab_diabetes = check_stability(df, "Has_Diabetes", features_no_age, model_type="rf")
    rf_stab_hyper = check_stability(df, "Has_Hypertension", features_no_age, model_type="rf")

    lr_all_results = pd.concat([lr_stab_diabetes, lr_stab_hyper], ignore_index=True)
    rf_all_results = pd.concat([rf_stab_diabetes, rf_stab_hyper], ignore_index=True)
    all_results = pd.concat([lr_all_results, rf_all_results], ignore_index=True)

    lr_all_results.to_csv(os.path.join(STABILITY_DIR, "feature_stability_lr.csv"), index=False)
    rf_all_results.to_csv(os.path.join(STABILITY_DIR, "feature_stability_rf.csv"), index=False)

    all_results.to_csv(os.path.join(STABILITY_DIR, "feature_stability_comparison.csv"), index=False)

    comparison_results = []
    for (race, target, feature), group in all_results.groupby(["Race", "Target", "Feature"]):
        lr_data = group[group["Model"] == "LR"]
        rf_data = group[group["Model"] == "RF"]
        
        if len(lr_data) > 0 and len(rf_data) > 0:
            comparison_results.append({
                "Race": race,
                "Target": target,
                "Feature": feature,
                "LR_Mean_Rank": lr_data["Mean_Rank"].iloc[0],
                "LR_Std_Rank": lr_data["Std_Rank"].iloc[0],
                "RF_Mean_Rank": rf_data["Mean_Rank"].iloc[0],
                "RF_Std_Rank": rf_data["Std_Rank"].iloc[0],
                "Rank_Diff": abs(lr_data["Mean_Rank"].iloc[0] - rf_data["Mean_Rank"].iloc[0]),
                "LR_Top_3_Freq": lr_data["Top_3_Freq"].iloc[0],
                "RF_Top_3_Freq": rf_data["Top_3_Freq"].iloc[0],
            })
    
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.sort_values(["Race", "Target", "Rank_Diff"], ascending=[True, True, False])
    comparison_df.to_csv(os.path.join(STABILITY_DIR, "feature_stability_comparison_summary.csv"), index=False)

    print("Top 3 Stable Features per Group (Logistic Regression)")
    lr_summary = lr_all_results.sort_values(["Race", "Target", "Mean_Rank"])
    for (race, target), group in lr_summary.groupby(["Race", "Target"]):
        print(f"\n{race} - {target}:")
        print(group[["Feature", "Mean_Rank", "Std_Rank", "Top_3_Freq"]].head(3).to_string(index=False))

    print("Top 3 Stable Features per Group (Random Forest)")
    rf_summary = rf_all_results.sort_values(["Race", "Target", "Mean_Rank"])
    for (race, target), group in rf_summary.groupby(["Race", "Target"]):
        print(f"\n{race} - {target}:")
        print(group[["Feature", "Mean_Rank", "Std_Rank", "Top_3_Freq"]].head(3).to_string(index=False))

    print("Features with Largest Rank Differences (LR vs RF)")
    for (race, target), group in comparison_df.groupby(["Race", "Target"]):
        print(f"\n{race} - {target}:")
        top_diff = group.head(5)
        print(top_diff[["Feature", "LR_Mean_Rank", "RF_Mean_Rank", "Rank_Diff"]].to_string(index=False))

if __name__ == "__main__":
    main()
