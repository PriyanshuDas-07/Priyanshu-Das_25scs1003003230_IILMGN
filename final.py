"""
Climate-Aware Breed Suitability Predictor
-----------------------------------------
USING ONLY XGBOOST
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.cluster import KMeans

from xgboost import XGBClassifier

# For class balancing
try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None
    print(
        "WARNING: imbalanced-learn not installed, SMOTE will be skipped.\n"
        "Install with: pip install imbalanced-learn"
    )

# =========================================================
# 0. CONSTANTS
# =========================================================

# Suitability label mapping
LABEL_MAP = {
    0: "Not suitable",
    1: "Moderately suitable",
    2: "Best fit",
}

# =========================================================
# 1. INPUT VALIDATION HELPERS
# =========================================================

def get_float_in_range(prompt, min_val, max_val):
    """Ask for a float and enforce [min_val, max_val]."""
    while True:
        raw = input(prompt)
        try:
            value = float(raw)
            if min_val <= value <= max_val:
                return value
            else:
                print(f"Value must be between {min_val} and {max_val}. Try again.")
        except ValueError:
            print(f"Invalid number: '{raw}'. Try again.")


def get_choice(prompt, valid_options):
    """Ask for a string that must be one of valid_options (case-insensitive)."""
    valid_lower = [opt.lower() for opt in valid_options]
    while True:
        value = input(prompt).strip().lower()
        if value in valid_lower:
            return value
        print(f"Invalid input. Allowed options: {', '.join(valid_lower)}")


# =========================================================
# 2. LOAD DATA
# =========================================================

def load_data():
    climate_df = pd.read_csv("climate_data.csv")
    region_df = pd.read_csv("region_data.csv")
    breed_df = pd.read_csv("breed_traits (1).csv")
    return climate_df, region_df, breed_df


# =========================================================
# 3. MERGE DATASETS
# =========================================================

def merge_datasets(climate_df, region_df, breed_df):
    # Merge climate + region
    region_climate = pd.merge(climate_df, region_df, on="region_id", how="inner")

    # Cross join with breed_df WITHOUT mutating originals
    region_climate_tmp = region_climate.assign(tmp_key=1)
    breed_tmp = breed_df.assign(tmp_key=1)

    full_df = (
        pd.merge(region_climate_tmp, breed_tmp, on="tmp_key")
        .drop(columns=["tmp_key"])
        .sort_values(by=["region_id", "breed_id"])
        .reset_index(drop=True)
    )

    return full_df


# =========================================================
# 4. RULE-BASED SUITABILITY LABELING
# =========================================================

def compute_suitability(row):
    score = 0

    temperature = row["temperature"]
    humidity = row["humidity"]
    rainfall = row["rainfall"]
    water_availability = str(row["water_availability"]).lower()
    disease_risk = str(row["disease_risk"]).lower()

    heat_tol = str(row["heat_tolerance"]).lower()
    cold_tol = str(row["cold_tolerance"]).lower()
    disease_res = str(row["disease_resistance"]).lower()
    feed_req = str(row["feed_requirement"]).lower()

    # ---- Temperature vs Tolerance ----
    if temperature >= 35:
        if heat_tol == "high":
            score += 2
        elif heat_tol == "medium":
            score += 1
    elif temperature <= 15:
        if cold_tol == "high":
            score += 2
        elif cold_tol == "medium":
            score += 1
    else:
        if heat_tol == "medium" or cold_tol == "medium":
            score += 1

    # ---- Humidity / Rainfall vs Disease Resistance ----
    if humidity >= 75 or rainfall >= 1500:
        if disease_res == "high":
            score += 2
        elif disease_res == "medium":
            score += 1
    else:
        if disease_res in ["medium", "high"]:
            score += 1

    # ---- Water Availability vs Feed Requirement ----
    if water_availability in ["high", "medium"]:
        if feed_req in ["medium", "high"]:
            score += 1
    else:
        if feed_req == "low":
            score += 2

    # ---- Disease risk vs disease_resistance ----
    if disease_risk == "high":
        if disease_res == "low":
            score -= 1
        elif disease_res == "high":
            score += 1

    # Map score to label
    if score <= 1:
        return 0  # Not suitable
    elif score == 2:
        return 1  # Moderately suitable
    else:
        return 2  # Best fit


def add_labels(full_df):
    full_df["suitability_label"] = full_df.apply(compute_suitability, axis=1)
    return full_df


# =========================================================
# 5. PREPROCESSING PIPELINE
# =========================================================

def build_preprocessor(df):
    # Columns we don't want as features
    drop_cols = ["breed_id", "breed_name", "region_id", "suitability_label"]

    # Feature columns = all except drop_cols (if they exist)
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Identify numeric and categorical columns
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    return preprocessor, numeric_cols, categorical_cols


# =========================================================
# 6. TRAIN / EVALUATE XGBOOST ONLY
# =========================================================

def train_xgboost_model(df):
    data = df.copy()

    # Drop rows with missing label if any
    data = data.dropna(subset=["suitability_label"])

    # Features / target
    X = data.drop(columns=["suitability_label"])
    y = data["suitability_label"]

    # Build preprocessor
    preprocessor, num_cols, cat_cols = build_preprocessor(data)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Fit preprocessor on train only
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Handle class imbalance using SMOTE (if available)
    if SMOTE is not None:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(
            X_train_processed, y_train
        )
    else:
        X_train_balanced, y_train_balanced = X_train_processed, y_train

    # Single model: XGBoost (probability output)
    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",  # probability output
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    print("\n" + "=" * 60)
    print("Training model: XGBoost")
    model.fit(X_train_balanced, y_train_balanced)

    # Use probabilities + argmax for predictions
    y_proba_test = model.predict_proba(X_test_processed)
    y_pred = np.argmax(y_proba_test, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0, target_names=[
        LABEL_MAP[0], LABEL_MAP[1], LABEL_MAP[2]
    ]))

    print("\n" + "#" * 60)
    print(f"Final model: XGBoost (macro F1={f1:.4f})")
    print("#" * 60)

    return model, preprocessor


# =========================================================
# 7. K-MEANS CLUSTERING FOR REGION PROFILING
# =========================================================

def cluster_regions(climate_df, region_df, n_clusters=4):
    # Merge climate + region
    rc = pd.merge(climate_df, region_df, on="region_id", how="inner")

    cluster_features = [
        "temperature",
        "humidity",
        "rainfall",
        "wind_speed",
        "altitude",
        "vegetation_index",
    ]

    rc_num = rc[cluster_features]

    scaler = StandardScaler()
    rc_scaled = scaler.fit_transform(rc_num)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    cluster_labels = kmeans.fit_predict(rc_scaled)

    rc["climate_cluster"] = cluster_labels

    # NOTE: labels 0–3 are arbitrary; names are just for readability
    profile_names = {
        0: "Dry-Heat Zone",
        1: "Humid Tropical",
        2: "Cool Highland",
        3: "Coastal Wet",
    }
    rc["climate_profile"] = rc["climate_cluster"].map(profile_names)

    return rc[["region_id", "climate_cluster", "climate_profile"]]


# =========================================================
# 8. RECOMMENDATION FUNCTION (REGION-BASED)
# =========================================================

def recommend_breeds_for_region(
    region_id,
    climate_df,
    region_df,
    breed_df,
    model,
    preprocessor,
    top_n=3
):
    # Get one region row
    climate_row = climate_df[climate_df["region_id"] == region_id]
    region_row = region_df[region_df["region_id"] == region_id]

    if climate_row.empty or region_row.empty:
        raise ValueError(f"Region_id {region_id} not found in climate/region data.")

    # Merge single region row
    region_features = pd.merge(climate_row, region_row, on="region_id")

    # Cross-join with all breeds without mutating breed_df
    region_features_tmp = region_features.assign(tmp_key=1)
    breed_tmp = breed_df.assign(tmp_key=1)

    candidate_df = (
        pd.merge(region_features_tmp, breed_tmp, on="tmp_key")
        .drop(columns=["tmp_key"])
    )

    # Save breed info for later
    breed_info = candidate_df[["breed_id", "breed_name"]].copy()

    # Drop any label columns if exist accidentally
    if "suitability_label" in candidate_df.columns:
        candidate_df = candidate_df.drop(columns=["suitability_label"])

    # Transform features
    X_processed = preprocessor.transform(candidate_df)

    # Get full class probabilities
    proba = model.predict_proba(X_processed)  # shape (n_samples, 3)

    # Attach probabilities with meaningful names
    breed_info["p_not_suitable"] = proba[:, 0]
    breed_info["p_moderately_suitable"] = proba[:, 1]
    breed_info["p_best_fit"] = proba[:, 2]

    # Use "Best fit" probability as ranking score
    breed_info["suitability_score"] = breed_info["p_best_fit"]

    # Sort descending and take top N
    top_breeds = breed_info.sort_values(
        by="suitability_score", ascending=False
    ).head(top_n)

    return top_breeds


# =========================================================
# 9. USER INPUT MODE WITH VALIDATION
# =========================================================

def get_user_input_and_recommend(model, preprocessor, climate_df, region_df, breed_df):
    print("\n---- USER INPUT PREDICTION MODE ----")
    print("\nEnter your climatic conditions:\n")

    # Validated numeric inputs
    temperature = get_float_in_range("Temperature (°C): ", -50, 60)
    humidity = get_float_in_range("Humidity (%): ", 0, 100)
    rainfall = get_float_in_range("Rainfall (mm/year): ", 0, 10000)
    wind_speed = get_float_in_range("Wind Speed (km/h): ", 0, 200)
    altitude = get_float_in_range("Altitude (m): ", -400, 9000)
    vegetation_index = get_float_in_range("Vegetation Index (0–1): ", 0, 1)

    # Validated categorical inputs
    soil_type = get_choice("Soil type (red/black/alluvial/sandy): ",
                           ["red", "black", "alluvial", "sandy"])
    season_trend = get_choice("Season trend (hot/cold/wet/dry): ",
                              ["hot", "cold", "wet", "dry"])
    water_availability = get_choice("Water availability (low/medium/high): ",
                                    ["low", "medium", "high"])
    disease_risk = get_choice("Disease risk (low/medium/high): ",
                              ["low", "medium", "high"])
    region_type = get_choice("Region type (urban/rural): ",
                             ["urban", "rural"])

    # Build a synthetic "region" row from user input
    custom_region = pd.DataFrame([{
        "region_id": 9999,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "wind_speed": wind_speed,
        "altitude": altitude,
        "vegetation_index": vegetation_index,
        "water_availability": water_availability,
        "disease_risk": disease_risk,
        "region_type": region_type,
        "soil_type": soil_type,
        "season_trend": season_trend,
    }])

    # Cross-join with all breeds (without mutating breed_df)
    custom_region_tmp = custom_region.assign(tmp_key=1)
    breed_tmp = breed_df.assign(tmp_key=1)
    candidate_df = (
        pd.merge(custom_region_tmp, breed_tmp, on="tmp_key")
        .drop(columns=["tmp_key"])
    )

    # Keep breed info
    breed_info = candidate_df[["breed_id", "breed_name"]].copy()

    # Drop label if present
    if "suitability_label" in candidate_df.columns:
        candidate_df = candidate_df.drop(columns=["suitability_label"])

    # Transform with same preprocessor used for training
    X_processed = preprocessor.transform(candidate_df)

    # Full class probabilities
    proba = model.predict_proba(X_processed)

    breed_info["p_not_suitable"] = proba[:, 0]
    breed_info["p_moderately_suitable"] = proba[:, 1]
    breed_info["p_best_fit"] = proba[:, 2]

    # Use "Best fit" probability as ranking key
    breed_info["suitability_score"] = breed_info["p_best_fit"]

    # Top N breeds (still showing all 3 class probabilities)
    top_n = 5
    top_results = breed_info.sort_values(
        by="suitability_score", ascending=False
    ).head(top_n)

    # Pretty printing with rounded probabilities
    print("\nTop Recommended Breeds for Your Climate:\n")
    display_df = top_results[[
        "breed_id",
        "breed_name",
        "p_not_suitable",
        "p_moderately_suitable",
        "p_best_fit",
    ]].copy()

    display_df.rename(columns={
        "p_not_suitable": f"P({LABEL_MAP[0]})",
        "p_moderately_suitable": f"P({LABEL_MAP[1]})",
        "p_best_fit": f"P({LABEL_MAP[2]})",
    }, inplace=True)

    # Format probabilities to 4 decimal places
    format_prob = lambda x: f"{x:.4f}"
    print(
        display_df.to_string(
            index=False,
            formatters={
                f"P({LABEL_MAP[0]})": format_prob,
                f"P({LABEL_MAP[1]})": format_prob,
                f"P({LABEL_MAP[2]})": format_prob,
            },
        )
    )
        # ★★★★★ Add Star Ratings and Bar Chart ★★★★★

    # Create star rating based on Best Fit Probability
    def star_rating(prob):
        if prob >= 0.9:
            return "★★★★★"
        elif prob >= 0.75:
            return "★★★★☆"
        elif prob >= 0.60:
            return "★★★☆☆"
        elif prob >= 0.40:
            return "★★☆☆☆"
        else:
            return "★☆☆☆☆"

    top_results["rating"] = top_results["p_best_fit"].apply(star_rating)

    print("\nRatings:")
    rating_df = top_results[["breed_name", "p_best_fit", "rating"]].copy()
    rating_df["p_best_fit"] = rating_df["p_best_fit"].apply(lambda x: f"{x:.2f}")
    print(rating_df.to_string(index=False))

    # Bar Chart Visualization
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.bar(top_results["breed_name"], top_results["p_best_fit"])
        plt.title("Best Fit Suitability Score Comparison")
        plt.ylabel("Probability of Best Fit")
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("\nVisualization failed:", e)
        print("Install matplotlib if missing: pip install matplotlib")



# =========================================================
# 10. MAIN SCRIPT
# =========================================================

def main():
    # 1. Load
    climate_df, region_df, breed_df = load_data()
    print("Loaded datasets:")
    print(f"  climate_data: {climate_df.shape}")
    print(f"  region_data:  {region_df.shape}")
    print(f"  breed_traits: {breed_df.shape}")

    # 2. Merge
    full_df = merge_datasets(climate_df, region_df, breed_df)
    print("\nMerged dataset shape (region x breed):", full_df.shape)

    # 3. Add rule-based suitability labels
    full_df = add_labels(full_df)
    print("Added suitability_label. Label distribution:")
    print(full_df["suitability_label"].value_counts().sort_index())
    print("\nLabel meaning:")
    for k, v in LABEL_MAP.items():
        print(f"  {k} -> {v}")

    # 4. Train XGBoost model
    best_model, preprocessor = train_xgboost_model(full_df)

    # 5. Cluster regions
    clusters_df = cluster_regions(climate_df, region_df, n_clusters=4)
    print("\nRegion climate clusters:")
    print(clusters_df.head())

    # 6. Example recommendation for a specific region
    example_region = full_df["region_id"].iloc[0]
    print(f"\nRecommending top breeds for region_id={example_region} ...")

    top_breeds = recommend_breeds_for_region(
        region_id=example_region,
        climate_df=climate_df,
        region_df=region_df,
        breed_df=breed_df,
        model=best_model,
        preprocessor=preprocessor,
        top_n=5,
    )

    print("\nTop recommended breeds for this region:")
    display_region_df = top_breeds[[
        "breed_id",
        "breed_name",
        "p_not_suitable",
        "p_moderately_suitable",
        "p_best_fit",
    ]].copy()

    display_region_df.rename(columns={
        "p_not_suitable": f"P({LABEL_MAP[0]})",
        "p_moderately_suitable": f"P({LABEL_MAP[1]})",
        "p_best_fit": f"P({LABEL_MAP[2]})",
    }, inplace=True)

    format_prob = lambda x: f"{x:.4f}"
    print(
        display_region_df.to_string(
            index=False,
            formatters={
                f"P({LABEL_MAP[0]})": format_prob,
                f"P({LABEL_MAP[1]})": format_prob,
                f"P({LABEL_MAP[2]})": format_prob,
            },
        )
    )

    # 7. User input mode
    get_user_input_and_recommend(best_model, preprocessor, climate_df, region_df, breed_df)


if __name__ == "__main__":
    main()
