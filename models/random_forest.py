# random forest from scratch, compared with sklearn's random forest

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# load the dataset
df = pd.read_csv("data/processed/car_new_details.csv")

# drop columns we don't want to use
df = df.drop(columns=["Model", "Location", "Owner", "Year", "Kilometer"])

# target is the price, features are everything else
X = df.drop(columns=["Price"])
y = df["Price"]

# text columns are categorical, number columns are numerical
categorical_columns = X.select_dtypes(exclude="number").columns.tolist()
numerical_columns = X.select_dtypes(include="number").columns.tolist()

# split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# one-hot encode the categories and keep the numbers as they are
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            categorical_columns,
        ),
        ("numerical", "passthrough", numerical_columns),
    ]
)

# learn the encoding from the training data, then apply it to both sets
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

print("Training data shape:", X_train_encoded.shape)
print("Testing data shape :", X_test_encoded.shape)


def evaluate_model(actual, predicted):
    # difference between real and predicted prices
    errors = actual - predicted

    mae = np.mean(np.abs(errors))  # average error
    mse = np.mean(errors ** 2)  # average squared error
    rmse = np.sqrt(mse)  # same as mse but in price units
    r2 = 1 - np.sum(errors ** 2) / np.sum((actual - np.mean(actual)) ** 2)

    print(f"MAE  : {mae:.2f}")
    print(f"MSE  : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.4f}")


class DecisionTree:
    def __init__(self, max_depth=6, min_samples_split=20,
                 max_features=None, rng=None):
        self.max_depth = max_depth  # how deep the tree can grow
        self.min_samples_split = min_samples_split  # smallest group we still split
        self.max_features = max_features  # how many features we try at each split
        self.rng = rng if rng is not None else np.random.RandomState()
        self.tree = None

    def _find_best_split(self, X, y, feature_indices):
        best_feature, best_threshold = None, None
        best_mse = float("inf")
        n = len(y)

        for f in feature_indices:
            values = X[:, f]
            unique_values = np.unique(values)

            # try every value if there are few, otherwise try 20 percentiles
            if len(unique_values) > 20:
                thresholds = np.unique(
                    np.percentile(values, np.linspace(5, 95, 20))
                )
            else:
                thresholds = unique_values

            for t in thresholds:
                # split the prices into a left group and a right group
                left_y = y[values <= t]
                right_y = y[values > t]

                # skip splits that leave one side empty
                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                # np.var is the mean squared error around the group's average
                weighted_mse = (
                    len(left_y) * np.var(left_y)
                    + len(right_y) * np.var(right_y)
                ) / n

                # keep the split with the lowest error
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = f
                    best_threshold = t

        return best_feature, best_threshold

    def _build_tree(self, X, y, depth=0):
        # stop growing: make a leaf that predicts the average price
        if (depth >= self.max_depth
                or len(y) < self.min_samples_split
                or np.all(y == y[0])):
            return {"leaf": True, "prediction": np.mean(y)}

        # pick a random set of features to try at this split
        feature_indices = self.rng.choice(
            X.shape[1], self.max_features, replace=False
        )

        feature, threshold = self._find_best_split(X, y, feature_indices)

        # no useful split found, so make a leaf
        if feature is None:
            return {"leaf": True, "prediction": np.mean(y)}

        # split the data and build the left and right branches
        left_mask = X[:, feature] <= threshold

        return {
            "leaf": False,
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X, y):
        # by default try all features at every split
        if self.max_features is None:
            self.max_features = X.shape[1]
        self.tree = self._build_tree(X, y)

    def _predict_row(self, node, row):
        # walk down the tree until we reach a leaf
        while not node["leaf"]:
            if row[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["prediction"]

    def predict(self, X):
        return np.array([self._predict_row(self.tree, row) for row in X])


class RandomForest:
    def __init__(self, n_trees=50, max_depth=6, min_samples_split=20,
                 max_features=None, random_state=42):
        self.n_trees = n_trees  # how many trees in the forest
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = np.random.RandomState(random_state)
        self.trees = []

    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_trees):
            # bootstrap sample: pick rows at random, the same row can repeat
            idx = self.rng.choice(len(X), size=len(X), replace=True)

            # train one tree on that sample
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                rng=self.rng,
            )
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)

    def predict(self, X):
        # the forest's prediction is the average of all the trees
        return np.mean([tree.predict(X) for tree in self.trees], axis=0)


n_features = X_train_encoded.shape[1]

# train our own random forest
print("\nTraining our Random Forest...")

our_model = RandomForest(
    n_trees=50,
    max_depth=6,
    min_samples_split=20,
    max_features=max(1, n_features // 3),  # a third of the features per split
    random_state=42,
)
our_model.fit(X_train_encoded, y_train)
our_predictions = our_model.predict(X_test_encoded)

# sklearn's forest with the same settings (fair comparison)
print("Training Sklearn Random Forest...")

sklearn_same = RandomForestRegressor(
    n_estimators=50,
    max_depth=6,
    min_samples_split=20,
    max_features=1 / 3,
    random_state=42,
    n_jobs=-1,
)
sklearn_same.fit(X_train_encoded, y_train)
sklearn_same_predictions = sklearn_same.predict(X_test_encoded)

# sklearn's forest with default settings (best case for reference)
print("Training sklearn Random Forest (defaults)...")

sklearn_default = RandomForestRegressor(
    n_estimators=100, random_state=42, n_jobs=-1
)
sklearn_default.fit(X_train_encoded, y_train)
sklearn_default_predictions = sklearn_default.predict(X_test_encoded)

# compare the three models on the test data
print("Random Forest Implementation from scratch")
evaluate_model(y_test, our_predictions)

print("Random Forest using Scikit-Learn from same settings")
evaluate_model(y_test, sklearn_same_predictions)

print("Random Forest using Scikit-Learn with it's default settings")
evaluate_model(y_test, sklearn_default_predictions)