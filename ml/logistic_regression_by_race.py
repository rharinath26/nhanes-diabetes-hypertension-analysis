import os
from typing import List, Dict
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

PARQUET_FOLDER = "parquet"
OUTPUT_DIR = "modeling_outputs"
LR_RACE_DIR = os.path.join(OUTPUT_DIR, "logistic_regression_race")
MODEL_DIR = os.path.join(OUTPUT_DIR, "logistic_regression_race", "saved_models")
os.makedirs(LR_RACE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

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

import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc

CURVE_DIR = os.path.join(LR_RACE_DIR, "curves")
os.makedirs(CURVE_DIR, exist_ok=True)

def sanitize_label(text: str) -> str:
    return (
        text.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
    )

def train_race_model(df, target, features):
    results = []
    
    for race_label, group_df in df.groupby("Race_Label"):
        if pd.isna(race_label): continue
        
        subset = group_df.dropna(subset=[target])
        if len(subset) < 200:
            print(f"Skipping {race_label}: too few samples ({len(subset)})")
            continue
            
        X = subset[features]
        y = subset[target].astype(int)
        
        if y.sum() < 10 or (len(y) - y.sum()) < 10:
            print(f"Skipping {race_label}: too few class samples")
            continue

        pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=1000, solver='lbfgs'))
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        try:
            probas = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
            auc_score = roc_auc_score(y, probas)
            ap_score = average_precision_score(y, probas)
            
            race_safe = sanitize_label(race_label)

            precision, recall, _ = precision_recall_curve(y, probas)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2, label=f'AP = {ap_score:.3f}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve - {target}\n{race_label} (Logistic Regression)')
            plt.legend(loc="lower left")
            pr_path = os.path.join(CURVE_DIR, f"{target.lower()}_{race_safe}_pr.png")
            plt.tight_layout()
            plt.savefig(pr_path, dpi=300, bbox_inches="tight")
            plt.close()

            fpr, tpr, _ = roc_curve(y, probas)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
            plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve - {target}\n{race_label} (Logistic Regression)')
            plt.legend(loc="lower right")
            roc_path = os.path.join(CURVE_DIR, f"{target.lower()}_{race_safe}_roc.png")
            plt.tight_layout()
            plt.savefig(roc_path, dpi=300, bbox_inches="tight")
            plt.close()
            
        except Exception as e:
            print(f"Error CV {race_label}: {e}")
            auc_score, ap_score = np.nan, np.nan

        pipeline.fit(X, y)
        model = pipeline.named_steps["model"]
        coeffs = model.coef_[0]

        model_path = os.path.join(MODEL_DIR, f"{target.lower()}_{race_safe}_pipeline.pkl")
        joblib.dump(pipeline, model_path)
        
        for feat, coef in zip(features, coeffs):
            results.append({
                "Race": race_label,
                "Target": target,
                "Feature": feat,
                "Readable_Feature": FEATURE_LABELS.get(feat, feat),
                "Coefficient": coef,
                "Abs_Coefficient": abs(coef),
                "ROC_AUC": auc_score,
                "PR_AUC": ap_score,
                "N_Samples": len(subset)
            })
            
    return pd.DataFrame(results)

def main():
    df = load_dataframe()
    features_no_age = [f for f in BASE_FEATURES if f not in ["RIDAGEYR", "RIDRETH3"]]
    
    print("\nRunning Race-Specific Logistic Regression (No Age)")
    
    res_diabetes = train_race_model(df, "Has_Diabetes", features_no_age)
    res_hyper = train_race_model(df, "Has_Hypertension", features_no_age)
    
    all_results = pd.concat([res_diabetes, res_hyper], ignore_index=True)

    all_results["Rank"] = all_results.groupby(["Race", "Target"])["Abs_Coefficient"].rank(ascending=False, method="first")
    
    out_path = os.path.join(LR_RACE_DIR, "lr_race_coefficients.csv")
    all_results.to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")

    print("\nTop 3 Features by Race (Logistic Regression):")
    summary = all_results[all_results["Rank"] <= 3].sort_values(["Target", "Race", "Rank"])
    print(summary[["Target", "Race", "Rank", "Readable_Feature", "Coefficient", "ROC_AUC"]].to_string(index=False))

if __name__ == "__main__":
    main()
