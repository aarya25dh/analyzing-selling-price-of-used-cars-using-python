# ============================================================
# PyTorch Neural Network - Used Car Price Prediction
# ============================================================

import pandas as pd
import numpy as np

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

torch.manual_seed(42)


# ============================================================
# 1. Load Dataset
# ============================================================

df = pd.read_csv("../data/processed/car_new_details.csv")


# ============================================================
# 2. Define Target and Features
# ============================================================

y = df["Price"]

# Drop Model and Location
# Also drop Car_Age (it is just the opposite of Year)
X = df.drop(
    columns=[
        "Price",
        "Model",
        "Location",
        "Car_Age"
    ]
)


# ============================================================
# 3. Define Feature Groups
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
    "Kilometers_Per_Year",
    "Owner_Count"
]

binary_columns = [
    "Unregistered"
]


# ============================================================
# 4. Train-Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 5. Preprocessing
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
            OneHotEncoder(handle_unknown="ignore", drop="first"),
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
# 6. Fit Preprocessor ONLY on Training Data
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# Convert sparse matrix to dense array
X_train_processed = X_train_processed.toarray()
X_test_processed = X_test_processed.toarray()


# ============================================================
# 7. Prepare Target (log + standardize)
# ============================================================

y_train_numpy = y_train.to_numpy(dtype=float)
y_test_numpy = y_test.to_numpy(dtype=float)

# Log makes the skewed prices more even
y_train_log = np.log1p(y_train_numpy)

# Standardize using TRAINING data only
y_mean = y_train_log.mean()
y_std = y_train_log.std()

y_train_scaled = (y_train_log - y_mean) / y_std


# ============================================================
# 8. Convert Data to PyTorch Tensors
# ============================================================

X_train_tensor = torch.tensor(
    X_train_processed,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_processed,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train_scaled,
    dtype=torch.float32
).reshape(-1, 1)


# ============================================================
# 9. Define Neural Network
# ============================================================

class CarPriceNN(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            # Input layer → Hidden layer
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Hidden layer
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Hidden layer
            nn.Linear(64, 32),
            nn.ReLU(),

            # Output layer
            # One output because Price is a regression target
            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# 10. Create Model
# ============================================================

input_size = X_train_tensor.shape[1]

model = CarPriceNN(input_size)


# ============================================================
# 11. Loss Function and Optimizer
# ============================================================

# MSE is commonly used for regression
criterion = nn.MSELoss()

# Adam optimizer (weight_decay helps avoid overfitting)
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-3
)


# ============================================================
# 12. Training
# ============================================================

epochs = 300

training_losses = []

for epoch in range(epochs):

    # Training mode
    model.train()

    # Forward pass
    predictions = model(X_train_tensor)

    # Calculate loss
    loss = criterion(
        predictions,
        y_train_tensor
    )

    # Clear previous gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update weights
    optimizer.step()

    # Store loss
    training_losses.append(loss.item())

    # Display progress
    if (epoch + 1) % 30 == 0:

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )


# ============================================================
# 13. Prediction (convert back to real price)
# ============================================================

model.eval()

with torch.no_grad():

    y_pred_tensor = model(X_test_tensor)


# Convert PyTorch tensor → NumPy
y_pred_scaled = y_pred_tensor.numpy().flatten()

y_pred = np.expm1(y_pred_scaled * y_std + y_mean)


# ============================================================
# 14. Evaluation
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
# 15. Display Results
# ============================================================

print("\n" + "=" * 60)
print("PyTorch Neural Network - Used Car Price Prediction")
print("=" * 60)

print(f"MAE : {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²  : {r2:.4f}")

print("=" * 60)

print("\nDataset Information:")
print(f"Total samples       : {len(df)}")
print(f"Training samples    : {len(X_train)}")
print(f"Testing samples     : {len(X_test)}")
print(f"Processed features  : {input_size}")
print(f"Epochs              : {epochs}")
print(f"Learning rate       : 0.001")

print("=" * 60)