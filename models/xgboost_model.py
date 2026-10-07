import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("data/processed/car_new_details.csv")

df = df.drop(columns=["Model", "Location", "Owner", "Year", "Kilometer"])

X = df.drop(columns=["Price"])
y = df["Price"]


# ============================================================
# 2. COLUMNS
# ============================================================

categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
numerical_columns = X.select_dtypes(include="number").columns.tolist()


# ============================================================
# 3. TRAIN-TEST SPLIT (same split as random_forest.py)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# 4. ONE-HOT ENCODING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical",
         OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         categorical_columns),
        ("numerical", "passthrough", numerical_columns),
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

print("Training data shape:", X_train_encoded.shape)
print("Testing data shape :", X_test_encoded.shape)


# ============================================================
# 5. EVALUATION
# ============================================================

def evaluate_model(actual, predicted):

    errors = actual - predicted

    mae = np.mean(np.abs(errors))
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    r2 = 1 - np.sum(errors ** 2) / np.sum((actual - np.mean(actual)) ** 2)

    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")


# ============================================================
# 6. XGBOOST MODEL
# ============================================================

def make_xgb():
    return XGBRegressor(
        n_estimators=500,       # number of boosting rounds (trees)
        learning_rate=0.05,     # how much each tree contributes
        max_depth=4,            # depth of each tree (shallow works well)
        subsample=0.8,          # use 80% of rows per tree
        colsample_bytree=0.8,   # use 80% of features per tree
        random_state=42,
        n_jobs=-1,
    )


print("\nTraining XGBoost...")

xgb_model = make_xgb()
xgb_model.fit(X_train_encoded, y_train)
xgb_predictions = xgb_model.predict(X_test_encoded)


# ============================================================
# 7. EVALUATE ON THE TEST SET
# ============================================================

print("\n======================================")
print("XGBOOST (test set)")
print("======================================")
evaluate_model(y_test, xgb_predictions)


# ============================================================
# 8. 5-FOLD CROSS-VALIDATION (more reliable than one split)
# ============================================================

print("\nRunning 5-fold cross-validation...")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", make_xgb()),
])

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

r2_scores = cross_val_score(pipeline, X, y, cv=kfold, scoring="r2")
mae_scores = -cross_val_score(
    pipeline, X, y, cv=kfold, scoring="neg_mean_absolute_error"
)

print("\n======================================")
print("XGBOOST (5-fold cross-validation)")
print("======================================")
print(f"R²  : {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
print(f"MAE : {mae_scores.mean():.2f}")


# ============================================================
# 9. TOP 10 MOST IMPORTANT FEATURES
# ============================================================

feature_names = preprocessor.get_feature_names_out()
importances = pd.Series(xgb_model.feature_importances_, index=feature_names)

print("\n======================================")
print("TOP 10 FEATURES")
print("======================================")
print(importances.sort_values(ascending=False).head(10).round(4))