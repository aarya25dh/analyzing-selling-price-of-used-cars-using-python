# xgboost model to predict used car prices

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# loading the dataset
df = pd.read_csv("data/processed/car_new_details.csv")

# dropping columns that are not needed for the model
df = df.drop(
    columns=["Model", "Location", "Owner", "Year"],
    errors="ignore",
)

# target is the price, features are everything else
X = df.drop(columns=["Price"])
y = df["Price"]

# text columns are categorical, number columns are numerical
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
numerical_columns = X.select_dtypes(include="number").columns.tolist()

print("Categorical columns:")
print(categorical_columns)

print("\nNumerical columns:")
print(numerical_columns)

# split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining rows:", len(X_train))
print("Testing rows :", len(X_test))

# one-hot encode the categories and keep the numbers as they are
encoder = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_columns,
        ),
        ("numerical", "passthrough", numerical_columns),
    ]
)

# learn the encoding from the training data only
X_train_encoded = encoder.fit_transform(X_train)

# apply the same encoding to the test data
X_test_encoded = encoder.transform(X_test)

print("\nEncoded training shape:", X_train_encoded.shape)
print("Encoded testing shape :", X_test_encoded.shape)

# creating the xgboost model
model = XGBRegressor(
    n_estimators=500,  # number of trees
    learning_rate=0.05,  # how much each tree adds
    max_depth=4,  # how deep each tree can grow
    subsample=0.8,  # each tree sees 80% of the rows
    colsample_bytree=0.8,  # each tree sees 80% of the features
    random_state=42,
)

# training the model
print("\nTraining XGBoost...")
model.fit(X_train_encoded, y_train)

# predicting prices for the test data
y_pred = model.predict(X_test_encoded)

# measure how good the predictions are
errors = y_test.to_numpy() - y_pred

mae = np.mean(np.abs(errors))  # average error
mse = np.mean(errors ** 2)  # average squared error
rmse = np.sqrt(mse) 
r2 = 1 - (
    np.sum(errors ** 2)
    / np.sum((y_test.to_numpy() - np.mean(y_test)) ** 2)
)

print("Xgboost Results")
print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R²  :", round(r2, 4))

# which features the model used the most to see correlation between features and price
feature_names = encoder.get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_,
})

importance = importance.sort_values(by="Importance", ascending=False)

print("The Top 10 Most Important Features for Predicting Used Car Prices are:")
print(importance.head(10).to_string(index=False))