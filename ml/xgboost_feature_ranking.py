import os
import joblib
import pandas as pd

MODEL_DIR = os.path.join("modeling_outputs", "saved_models")
OUTPUT_DIR = os.path.join("modeling_outputs", "xgboost_feature_rankings")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_rankings(model_path: str) -> pd.DataFrame:
    pipeline = joblib.load(model_path)
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        raise ValueError(f"Model in {model_path} lacks feature_importances_.")

    imputer = pipeline.named_steps["imputer"]
    feature_names = getattr(imputer, "feature_names_in_", None)
    if feature_names is None:
        raise ValueError(f"No feature names recorded for {model_path}.")

    df = pd.DataFrame(
        {"Feature": feature_names, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)
    df["Importance_Normalized"] = df["Importance"] / df["Importance"].sum()
    df["Rank"] = df["Importance"].rank(method="first", ascending=False).astype(int)
    df["Model_File"] = os.path.basename(model_path)
    return df


def main():
    model_files = [
        os.path.join(MODEL_DIR, fname)
        for fname in os.listdir(MODEL_DIR)
        if fname.endswith(".pkl") and "xgb" in fname.lower()
    ]
    if not model_files:
        print(f"No XGBoost pipelines found in {MODEL_DIR}.")
        return

    summaries = []
    for path in sorted(model_files):
        try:
            ranking = load_rankings(path)
        except Exception as exc:
            print(f"Skipping {path}: {exc}")
            continue

        outfile = os.path.join(
            OUTPUT_DIR, f"{os.path.basename(path).replace('.pkl', '_ranked.csv')}"
        )
        ranking.to_csv(outfile, index=False)
        print(f"Saved {outfile}")
        summaries.append(ranking.assign(Output_File=os.path.basename(outfile)))

    if summaries:
        summary = pd.concat(summaries, ignore_index=True)
        summary_path = os.path.join(OUTPUT_DIR, "xgboost_feature_rankings_all.csv")
        summary.to_csv(summary_path, index=False)
        print(f"Combined ranking written to {summary_path}")


if __name__ == "__main__":
    main()

