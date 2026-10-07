import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
class LinearRegression:
    def __init__(self, learning_rate=0.05, n_iterations=3000, alpha=10):
        self.alpha = alpha  
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
        
        y_hat = X @ self.weights + self.bias
        
        de_dw = -(2 / n) * np.dot(X.T, (y - y_hat))

        de_dw += (2 * self.alpha / n) * self.weights

        de_db = -(2 / n) * np.sum(y - y_hat)

        self.weights -= self.learning_rate * de_dw
        self.bias -= self.learning_rate * de_db

    def fit(self, X, y):

        #initialize weights
        self.weights = np.zeros(X.shape[1])
        self.loss = []

        for i in range(self.n_iterations):

            self._gradient_descent(X, y)

         
            y_hat = X @ self.weights + self.bias    #calculate predictions

            loss = self._mean_squared_error(y, y_hat)
            
            self.loss.append(loss)

            if (i + 1) % 300 == 0:
                print(
                    f"Iteration [{i + 1}/{self.n_iterations}] "
                    f"Loss: {loss:.4f}"
                )

    def predict(self, X):

        return X @ self.weights + self.bias


df = pd.read_csv("../data/processed/car_new_details.csv")

y = df["Price"]

X = df.drop(
    columns=[
        "Price",
        "Model",
        "Location",
        "Car_Age"
    ]
)

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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        #numerical to standardScaler
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),

        #categorical to One-Hot Encoding
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", drop="first"),
            categorical_columns
        ),

        #binary to Keep 0/1
        (
            "binary",
            "passthrough",
            binary_columns
        )
    ]
)


X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

#convert sparse matrix to dense NumPy array
X_train_processed = X_train_processed.toarray()
X_test_processed = X_test_processed.toarray()


y_train_numpy = y_train.to_numpy(dtype=float)
y_test_numpy = y_test.to_numpy(dtype=float)

y_train_log = np.log1p(y_train_numpy)

y_mean = y_train_log.mean()
y_std = y_train_log.std()

y_train_scaled = (y_train_log - y_mean) / y_std


model = LinearRegression(
    learning_rate=0.05,
    n_iterations=3000,
    alpha=10
)

model.fit(
    X_train_processed,
    y_train_scaled
)


y_pred_scaled = model.predict(X_test_processed)

y_pred = np.expm1(y_pred_scaled * y_std + y_mean)

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
print(f"L2 alpha            : {model.alpha}")

print("=" * 60)