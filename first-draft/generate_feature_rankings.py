import os
import joblib
import pandas as pd

MODEL_DIR = os.path.join("modeling_outputs", "saved_models")
OUTPUT_DIR = os.path.join("modeling_outputs", "feature_rankings")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_feature_ranking(model_path: str) -> pd.DataFrame:
    pipeline = joblib.load(model_path)
    imputer = pipeline.named_steps["imputer"]
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"Model in {model_path} does not expose feature_importances_.")

    feature_names = getattr(imputer, "feature_names_in_", None)
    if feature_names is None:
        raise ValueError(
            f"Could not determine feature names for {model_path}; "
            "ensure the pipeline was fit after scikit-learn 1.0."
        )

    df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    ).sort_values("Importance", ascending=False)
    df["Importance_Normalized"] = df["Importance"] / df["Importance"].sum()
    df["Rank"] = df["Importance"].rank(method="first", ascending=False).astype(int)
    df["Model_File"] = os.path.basename(model_path)
    return df


def main():
    model_files = [
        os.path.join(MODEL_DIR, fname)
        for fname in os.listdir(MODEL_DIR)
        if fname.endswith(".pkl")
    ]
    if not model_files:
        print(f"No saved models found in {MODEL_DIR}.")
        return

    all_rankings = []
    for path in sorted(model_files):
        try:
            ranking = load_feature_ranking(path)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue

        outfile = os.path.join(
            OUTPUT_DIR, f"{os.path.basename(path).replace('.pkl', '_ranked.csv')}"
        )
        ranking.to_csv(outfile, index=False)
        print(f"Saved feature ranking: {outfile}")
        all_rankings.append(ranking.assign(Output_File=os.path.basename(outfile)))

    if all_rankings:
        summary = pd.concat(all_rankings, ignore_index=True)
        summary_path = os.path.join(OUTPUT_DIR, "all_feature_rankings.csv")
        summary.to_csv(summary_path, index=False)
        print(f"\nCombined rankings saved to {summary_path}")


if __name__ == "__main__":
    main()

