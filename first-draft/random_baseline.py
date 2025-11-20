import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score

OUTPUT_DIR = os.path.join("modeling_outputs", "random_baseline")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MPL_CACHE_DIR = os.path.join(OUTPUT_DIR, ".mplcache")
os.makedirs(MPL_CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_CACHE_DIR)
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

PARQUET_FOLDER = "parquet"


def load_dataframe() -> pd.DataFrame:
    demo = pd.read_parquet(os.path.join(PARQUET_FOLDER, "DEMO_L.parquet"))
    diq = pd.read_parquet(os.path.join(PARQUET_FOLDER, "DIQ_L.parquet"))
    bpq = pd.read_parquet(os.path.join(PARQUET_FOLDER, "BPQ_L.parquet"))

    df = demo.copy()
    df = df.merge(diq[["SEQN", "DIQ010"]], on="SEQN", how="left")
    df = df.merge(bpq[["SEQN", "BPQ020"]], on="SEQN", how="left")

    df["Has_Diabetes"] = (df["DIQ010"] == 1).astype(float)
    df["Has_Diabetes"] = df["Has_Diabetes"].where(~df["DIQ010"].isin([7, 9]))
    df["Has_Hypertension"] = (df["BPQ020"] == 1).astype(float)
    df["Has_Hypertension"] = df["Has_Hypertension"].where(~df["BPQ020"].isin([7, 9]))
    return df.dropna(subset=["Has_Diabetes", "Has_Hypertension"])


def plot_baseline(y: np.ndarray, probs: np.ndarray, target: str):
    fpr, tpr, _ = roc_curve(y, probs)
    precision, recall, _ = precision_recall_curve(y, probs)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y, probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="gray", lw=2, label=f"Baseline ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="lightgray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Random Baseline ROC - {target}")
    plt.legend()
    roc_path = os.path.join(OUTPUT_DIR, f"{target.lower()}_random_roc.png")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color="gray", lw=2, label=f"Baseline PR (AP={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Random Baseline PR - {target}")
    plt.legend()
    pr_path = os.path.join(OUTPUT_DIR, f"{target.lower()}_random_pr.png")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=300)
    plt.close()

    print(f"{target}: random ROC-AUC={roc_auc:.3f}, PR-AUC={pr_auc:.3f}")


def main():
    df = load_dataframe()
    rng = np.random.default_rng(42)

    for target in ["Has_Diabetes", "Has_Hypertension"]:
        y = df[target].astype(int).values
        prevalence = y.mean()
        probs = rng.uniform(low=0.0, high=1.0, size=len(y))
        probs = (probs + prevalence) / 2.0
        plot_baseline(y, probs, target)

    print(f"Random baseline curves saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

