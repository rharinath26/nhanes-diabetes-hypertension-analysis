import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Directories
PERM_DIR = "ml/modeling_outputs/random_forest/permutation_importance"
STABILITY_DIR = "ml/modeling_outputs/random_forest/stability"
OUTPUT_DIR = "ml/modeling_outputs"

# Label mapping - Fix incorrect labels
# PAD680 = Sedentary Behavior (sitting time)
# PAD800 = Moderate Activity
LABEL_MAPPING = {
    "Moderate Activity (min/wk)": "Sedentary Behavior (min/day)",  # This was PAD680 mislabeled
    "Vigorous Activity (min/wk)": "Moderate Activity (min/wk)",    # This was PAD800 mislabeled
    "Mod_Activity": "Sedentary_Behavior",  # For short names (PAD680)
    "Vig_Activity": "Mod_Activity"         # For short names (PAD800)
}

# Reverse mapping to fix any already-swapped labels
REVERSE_MAPPING = {
    "Sedentary Behavior (min/day)": None,  # Keep as is if it's for PAD680
    "Moderate Activity (min/wk)": None,    # Keep as is if it's for PAD800
}

def update_csv_labels(csv_path):
    """Update labels in a CSV file based on Feature column"""
    df = pd.read_csv(csv_path)
    
    # Create a mapping based on the actual Feature column
    if 'Feature' in df.columns and 'Feature_Label' in df.columns:
        for idx, row in df.iterrows():
            feature = row['Feature']
            current_label = row['Feature_Label']
            
            # Fix PAD680 - should be Sedentary Behavior
            if feature == 'PAD680':
                if 'Moderate Activity' in current_label or 'Sedentary' not in current_label:
                    df.at[idx, 'Feature_Label'] = 'Sedentary Behavior (min/day)'
            
            # Fix PAD800 - should be Moderate Activity
            elif feature == 'PAD800':
                if 'Vigorous Activity' in current_label or 'Sedentary' in current_label:
                    df.at[idx, 'Feature_Label'] = 'Moderate Activity (min/wk)'
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)
    print(f"Updated: {csv_path}")
    return df

def regenerate_permutation_plot(csv_path):
    """Regenerate permutation importance plot from CSV"""
    df = pd.read_csv(csv_path)
    
    # Get top 15 features
    top_features = df.head(15)
    
    # Create plot
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
    
    # Extract target and label from filename
    filename = os.path.basename(csv_path)
    parts = filename.replace("perm_importance_", "").replace(".csv", "").split("_")
    target = parts[0] + "_" + parts[1]
    label = "_".join(parts[2:])
    
    model_type = "Random Forest" if "lr_" not in csv_path else "Logistic Regression"
    
    ax.set_title(
        f"Permutation Importance - {target.replace('has_', '').replace('_', ' ').title()} ({label.replace('_', ' ').title()})\nTop 15 Features ({model_type})",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    
    # Save plot
    plot_path = csv_path.replace(".csv", ".png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Regenerated plot: {plot_path}")

def main():
    print("Updating CSV labels and regenerating plots...")
    
    # Update permutation importance files
    if os.path.exists(PERM_DIR):
        print(f"\\nProcessing {PERM_DIR}...")
        for filename in os.listdir(PERM_DIR):
            if filename.endswith(".csv"):
                csv_path = os.path.join(PERM_DIR, filename)
                update_csv_labels(csv_path)
                regenerate_permutation_plot(csv_path)
    
    # Update LR permutation importance files
    lr_perm_dir = "ml/modeling_outputs/lr_permutation_importance"
    if os.path.exists(lr_perm_dir):
        print(f"\\nProcessing {lr_perm_dir}...")
        for filename in os.listdir(lr_perm_dir):
            if filename.endswith(".csv"):
                csv_path = os.path.join(lr_perm_dir, filename)
                update_csv_labels(csv_path)
                regenerate_permutation_plot(csv_path)
    
    # Update stability files
    if os.path.exists(STABILITY_DIR):
        print(f"\\nProcessing {STABILITY_DIR}...")
        for filename in os.listdir(STABILITY_DIR):
            if filename.endswith(".csv"):
                csv_path = os.path.join(STABILITY_DIR, filename)
                update_csv_labels(csv_path)
    
    # Update overall feature importance files
    print(f"\\nProcessing overall feature importance files...")
    for filename in os.listdir(OUTPUT_DIR):
        if "feature_importance" in filename and filename.endswith(".csv"):
            csv_path = os.path.join(OUTPUT_DIR, filename)
            update_csv_labels(csv_path)
    
    print("\\nDone! All CSV files updated and plots regenerated.")

if __name__ == "__main__":
    main()
