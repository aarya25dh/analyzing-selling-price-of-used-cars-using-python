# linear regression from scratch using gradient descent algorithm to predict used car prices
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class LinearRegression:
    def __init__(self, learning_rate=0.05, n_iterations=3000, alpha=10):
        self.alpha = alpha  # strength of the L2 penalty (keeps weights small)
        self.learning_rate = learning_rate  # size of each update step
        self.n_iterations = n_iterations  # how many times we update
        self.weights = None
        self.bias = 0
        self.loss = []  # loss after every iteration

    @staticmethod
    def _mean_squared_error(y, y_hat):
        # average of the squared differences
        n = y.shape[0]
        return np.sum((y - y_hat) ** 2) / n

    def _gradient_descent(self, X, y):
        n = y.shape[0]

        # current predictions: y_hat = X * w + b
        y_hat = X @ self.weights + self.bias

        # calculating the slope of the loss for the weights
        de_dw = -(2 / n) * np.dot(X.T, (y - y_hat))

        # adding the L2 penalty to the weight slope
        de_dw += (2 * self.alpha / n) * self.weights

        # calculating the slope of the loss for the bias
        de_db = -(2 / n) * np.sum(y - y_hat)

        # moving weights and bias using learning rate to reduce it slowly 
        self.weights -= self.learning_rate * de_dw
        self.bias -= self.learning_rate * de_db

    def fit(self, X, y):
        # starting with all weights at zero
        self.weights = np.zeros(X.shape[1])
        self.loss = []

        for i in range(self.n_iterations):
            self._gradient_descent(X, y)

            # track the loss so we can see it going down
            y_hat = X @ self.weights + self.bias
            loss = self._mean_squared_error(y, y_hat)
            self.loss.append(loss)

            # printing progress every 300 iterations
            if (i + 1) % 300 == 0:
                print(f"Iteration [{i + 1}/{self.n_iterations}] Loss: {loss:.4f}")

    def predict(self, X):
        return X @ self.weights + self.bias


# loading the dataset from processed data 
df = pd.read_csv("data/processed/car_new_details.csv")

# target is the price, features are everything else
y = df["Price"]
X = df.drop(columns=["Price", "Model", "Location", "Car_Age"])

# grouping the columns by how they need to be prepared
# categorical columns are text so we encode it and numerical columns are numbers so we scale it
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

# splitting into 80% training and 20% testing
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

# create the model and train it
model = LinearRegression(learning_rate=0.05, n_iterations=3000, alpha=10)
model.fit(X_train_processed, y_train_scaled)

# predict on the test data, then undo the scaling and log to get real prices
y_pred_scaled = model.predict(X_test_processed)
y_pred = np.expm1(y_pred_scaled * y_std + y_mean)

# measure how good the predictions are
mae = mean_absolute_error(y_test_numpy, y_pred)
rmse = mean_squared_error(y_test_numpy, y_pred) ** 0.5
r2 = r2_score(y_test_numpy, y_pred)

# results
print("Linear Regression for Used Car Price Prediction")
print(f"MAE : {mae:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²  : {r2:.4f}")

print("\nDataset Information:")
print(f"Total samples       : {len(df)}")
print(f"Training samples    : {len(X_train)}")
print(f"Testing samples     : {len(X_test)}")
print(f"Processed features  : {X_train_processed.shape[1]}")
print(f"Learning rate       : {model.learning_rate}")
print(f"Iterations          : {model.n_iterations}")
print(f"L2 alpha            : {model.alpha}")