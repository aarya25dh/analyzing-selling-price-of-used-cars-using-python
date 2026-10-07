# comparing the four models on the same train/test split
import os

# this is for not crashing when xgboost and pytorch are run together
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from xgboost import XGBRegressor

# loading the dataset and dropping columns that are not needed for the models
df = pd.read_csv("data/processed/car_new_details.csv")
df = df.drop(columns=["Model", "Location", "Owner", "Year"])

# target is the price and the features are everything else
y = df["Price"]
X = df.drop(columns=["Price"])

# splitting the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# text columns are categorical, number columns are numerical
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
numerical_columns = X.select_dtypes(include="number").columns.tolist()


def prepare_data(scale_numbers, min_frequency=None):
    # one-hot encoding the text categories
    if min_frequency is None:
        unknown_option = "ignore"
    else:
        unknown_option = "infrequent_if_exist"

    encoder = OneHotEncoder(
        handle_unknown=unknown_option,
        min_frequency=min_frequency,
        sparse_output=False,
    )

    # scaling the numbers if the model needs it like linear regression and neural networks
    if scale_numbers:
        number_step = StandardScaler()
    else:
        number_step = "passthrough"

    preprocessor = ColumnTransformer([
        ("categorical", encoder, categorical_columns),
        ("numerical", number_step, numerical_columns),
    ])

    # learning from the training data only, then apply to both
    train = np.asarray(preprocessor.fit_transform(X_train), dtype=float)
    test = np.asarray(preprocessor.transform(X_test), dtype=float)
    return train, test


# random forest and xgboost do not need scaled numbers
X_train_tree, X_test_tree = prepare_data(scale_numbers=False)

# the neural network needs scaled numbers so scaling here
X_train_scaled, X_test_scaled = prepare_data(scale_numbers=True)

# plain linear regression gets confused by rare categories so categories with fewer than 20 cars are grouped together
X_train_linear, X_test_linear = prepare_data(scale_numbers=True, min_frequency=20)

# linear regression and neural network learn better from log(price) so we also scale it but only on the training data
y_train_log = np.log1p(y_train.to_numpy(dtype=float))
y_mean = y_train_log.mean()
y_std = y_train_log.std()
y_train_scaled = (y_train_log - y_mean) / y_std


def to_real_price(predictions_scaled):
    # redo the scaling to get the actual prices
    return np.expm1(predictions_scaled * y_std + y_mean)


# the predictions of every model are stored here
predictions = {}

# 1. linear regression
print("Training Linear Regression...")
linear_model = LinearRegression()
linear_model.fit(X_train_linear, y_train_scaled)
predictions["Linear Regression"] = to_real_price(linear_model.predict(X_test_linear))

# 2. random forest
print("Training Random Forest...")
forest_model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
forest_model.fit(X_train_tree, y_train)
predictions["Random Forest"] = forest_model.predict(X_test_tree)

# 3. xgboost
print("Training XGBoost...")
xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)
xgb_model.fit(X_train_tree, y_train)
predictions["XGBoost"] = xgb_model.predict(X_test_tree)

# 4. neural network (pytorch)
print("Training Neural Network...")
torch.manual_seed(42)
torch.set_num_threads(1)  # avoids the crash on Mac

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).reshape(-1, 1)

network = nn.Sequential(
    nn.Linear(X_train_tensor.shape[1], 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=0.001, weight_decay=1e-3)

# training for 300 rounds
for epoch in range(300):
    network.train()
    loss = loss_function(network(X_train_tensor), y_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# predicting for the test data
network.eval()
with torch.no_grad():
    network_output = network(X_test_tensor).numpy().flatten()
predictions["Neural Network"] = to_real_price(network_output)

# measuring the errors of every model
rows = []
for name, predicted_prices in predictions.items():
    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predicted_prices),
        "RMSE": mean_squared_error(y_test, predicted_prices) ** 0.5,
        "R2": r2_score(y_test, predicted_prices),
    })

# show the results, best model first
results = pd.DataFrame(rows).sort_values("R2", ascending=False)

print("\n" + "=" * 55)
print("MODEL COMPARISON")
print("=" * 55)
print(results.to_string(
    index=False,
    formatters={
        "MAE": "{:,.0f}".format,
        "RMSE": "{:,.0f}".format,
        "R2": "{:.4f}".format,
    },
))
print("=" * 55)