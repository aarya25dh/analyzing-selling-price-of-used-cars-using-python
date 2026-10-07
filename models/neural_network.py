# pytorch neural network to predict used car prices

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

# random seed value is 42
torch.manual_seed(42)

# load the dataset (run this file from the project root folder)
df = pd.read_csv("data/processed/car_new_details.csv")

# target is the price, features are everything else
# Model and Location have too many values, Car_Age is just the opposite of Year
y = df["Price"]
X = df.drop(columns=["Price", "Model", "Location", "Car_Age"])

# group the columns by how they need to be prepared
categorical_columns = [
    "Make",
    "Fuel Type",
    "Transmission",
    "Color",
    "Seller Type",
    "Drivetrain",
    "Location_Grouped",
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
    "Owner_Count",
]

binary_columns = ["Unregistered"]

# split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# numbers are scaled, categories are one-hot encoded, 0/1 columns are kept as they are
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            categorical_columns,
        ),
        ("binary", "passthrough", binary_columns),
    ]
)

# learn the scaling and encoding from the training data only
X_train_processed = preprocessor.fit_transform(X_train)

# apply the same scaling and encoding to the test data
X_test_processed = preprocessor.transform(X_test)

y_train_numpy = y_train.to_numpy(dtype=float)
y_test_numpy = y_test.to_numpy(dtype=float)

# log makes the skewed prices more even
y_train_log = np.log1p(y_train_numpy)

# scale the target using training data only
y_mean = y_train_log.mean()
y_std = y_train_log.std()
y_train_scaled = (y_train_log - y_mean) / y_std

# convert the data to pytorch tensors
X_train_tensor = torch.tensor(X_train_processed, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).reshape(-1, 1)


class CarPriceNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.network = nn.Sequential(
            # input layer to first hidden layer
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # randomly switch off 20% of neurons to avoid overfitting

            # second hidden layer
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            # third hidden layer
            nn.Linear(64, 32),
            nn.ReLU(),

            # output layer: one number because we predict one price
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)


# create the model
input_size = X_train_tensor.shape[1]
model = CarPriceNN(input_size)

# mean squared error is a common loss for predicting numbers
criterion = nn.MSELoss()

# adam updates the weights, weight_decay helps avoid overfitting
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-3,
)

# train the model
epochs = 300
training_losses = []

for epoch in range(epochs):
    # switch to training mode
    model.train()

    # predict, then measure how wrong the predictions are
    predictions = model(X_train_tensor)
    loss = criterion(predictions, y_train_tensor)

    # clear old gradients, calculate new ones, then update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    training_losses.append(loss.item())

    # print progress every 30 epochs
    if (epoch + 1) % 30 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}] Loss: {loss.item():.4f}")

# switch to evaluation mode (turns dropout off)
model.eval()

# predict on the test data without tracking gradients
with torch.no_grad():
    y_pred_tensor = model(X_test_tensor)

# convert to numpy, then undo the scaling and log to get real prices
y_pred_scaled = y_pred_tensor.numpy().flatten()
y_pred = np.expm1(y_pred_scaled * y_std + y_mean)

# measure how good the predictions are
mae = mean_absolute_error(y_test_numpy, y_pred)
rmse = mean_squared_error(y_test_numpy, y_pred) ** 0.5
r2 = r2_score(y_test_numpy, y_pred)

# show the results
print("Neural Network using PyTorch for Used Car Price Prediction")
print(f"MAE : {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²  : {r2:.4f}")

print("\nDataset Information:")
print(f"Total samples       : {len(df)}")
print(f"Training samples    : {len(X_train)}")
print(f"Testing samples     : {len(X_test)}")
print(f"Processed features  : {input_size}")
print(f"Epochs              : {epochs}")
print("Learning rate       : 0.001")