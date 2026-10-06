# ============================================================
# Linear Regression - Used Car Price Prediction
# Custom Gradient Descent Implementation
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. Custom Linear Regression
# ============================================================

class LinearRegression:
    def __init__(self, learning_rate=0.05, n_iterations=2000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0
        self.loss = []

    @staticmethod
    def _mean_squared_error(y, y_hat):
        n = y.shape[0]
        error = np.sum((y - y_hat) ** 2) / n
        return error

    def _gradient_descent(self, X, y):

        n = y.shape[0]

        # Prediction
        # Correct formula for multiple features:
        # y_hat = Xw + b
        y_hat = X @ self.weights + self.bias

        # Gradient for weights
        de_dw = -(2 / n) * np.dot(X.T, (y - y_hat))

        # Gradient for bias
        de_db = -(2 / n) * np.sum(y - y_hat)

        # Update weights and bias
        self.weights -= self.learning_rate * de_dw
        self.bias -= self.learning_rate * de_db

    def fit(self, X, y):

        # Initialize weights
        self.weights = np.zeros(X.shape[1])

        for i in range(self.n_iterations):

            self._gradient_descent(X, y)

            # Calculate predictions
            y_hat = X @ self.weights + self.bias

            # Calculate loss
            loss = self._mean_squared_error(y, y_hat)

            self.loss.append(loss)

            if (i + 1) % 100 == 0:
                print(
                    f"Iteration [{i + 1}/{self.n_iterations}] "
                    f"Loss: {loss:,.2f}"
                )

    def predict(self, X):

        return X @ self.weights + self.bias


# ============================================================
# 2. Load Dataset
# ============================================================

df = pd.read_csv("../data/processed/car_new_details.csv")


# ============================================================
# 3. Define Target and Features
# ============================================================

# Target
# IMPORTANT:
# Price is NOT scaled or transformed.
y = df["Price"]

# Drop Model and Location
X = df.drop(
    columns=[
        "Price",
        "Model",
        "Location"
    ]
)


# ============================================================
# 4. Define Feature Groups
# ============================================================

categorical_columns = [
    "Make",
    "Fuel Type",
    "Transmission",
    "Color",
    "Seller Type",
    "Drivetrain",
    "Location_Grouped"
]

numerical_columns = [
    "Year",
    "Kilometer",
    "Engine",
    "Max Power",
    "Max Torque",
    "Length",
    "Width",
    "Height",
    "Seating Capacity",
    "Fuel Tank Capacity",
    "Car_Age",
    "Kilometers_Per_Year",
    "Owner_Count"
]

binary_columns = [
    "Unregistered"
]


# ============================================================
# 5. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 6. Preprocessing
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        # Numerical → StandardScaler
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),

        # Categorical → One-Hot Encoding
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        ),

        # Binary → Keep 0/1
        (
            "binary",
            "passthrough",
            binary_columns
        )
    ]
)


# ============================================================
# 7. Fit Preprocessor ONLY on Training Data
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# OneHotEncoder produces a sparse matrix.
# Convert it to a dense NumPy array for our custom model.

X_train_processed = X_train_processed.toarray()
X_test_processed = X_test_processed.toarray()


# Convert target to NumPy
y_train_numpy = y_train.to_numpy(dtype=float)
y_test_numpy = y_test.to_numpy(dtype=float)


# ============================================================
# 8. Create and Train Model
# ============================================================

model = LinearRegression(
    learning_rate=0.05,
    n_iterations=2000
)

model.fit(
    X_train_processed,
    y_train_numpy
)


# ============================================================
# 9. Make Predictions
# ============================================================

y_pred = model.predict(X_test_processed)


# ============================================================
# 10. Evaluation
# ============================================================

mae = mean_absolute_error(
    y_test_numpy,
    y_pred
)

rmse = mean_squared_error(
    y_test_numpy,
    y_pred
) ** 0.5

r2 = r2_score(
    y_test_numpy,
    y_pred
)


# ============================================================
# 11. Display Results
# ============================================================

print("\n" + "=" * 60)
print("Custom Linear Regression - Used Car Price Prediction")
print("=" * 60)

print(f"MAE : {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²  : {r2:.4f}")

print("=" * 60)

print("\nDataset Information:")
print(f"Total samples       : {len(df)}")
print(f"Training samples    : {len(X_train)}")
print(f"Testing samples     : {len(X_test)}")
print(f"Processed features  : {X_train_processed.shape[1]}")
print(f"Learning rate       : {model.learning_rate}")
print(f"Iterations          : {model.n_iterations}")

print("=" * 60)
