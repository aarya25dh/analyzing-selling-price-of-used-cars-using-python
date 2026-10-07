import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
torch.manual_seed(42)

df = pd.read_csv("../data/processed/car_new_details.csv")

#Target and Features

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



#Train-Test Split


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

preprocessor = ColumnTransformer(
    transformers=[
        #StandardScaler
        (
            "num",
            StandardScaler(),
            numerical_columns
        ),

        #One-Hot Encoding
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", drop="first"),
            categorical_columns
        ),

       #Keep 0/1
        (
            "binary",
            "passthrough",
            binary_columns
        )
    ]
)



X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


#matrix to dense array
X_train_processed = X_train_processed.toarray()
X_test_processed = X_test_processed.toarray()

#Prepare Target
y_train_numpy = y_train.to_numpy(dtype=float)
y_test_numpy = y_test.to_numpy(dtype=float)

y_train_log = np.log1p(y_train_numpy)


y_mean = y_train_log.mean()
y_std = y_train_log.std()

y_train_scaled = (y_train_log - y_mean) / y_std



#Convert Data to PyTorch Tensors

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

#Define Neural Network
class CarPriceNN(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            #Input layer
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
            nn.Linear(32, 1)
        )

    def forward(self, x):

        return self.network(x)



#Create Model
input_size = X_train_tensor.shape[1]

model = CarPriceNN(input_size)



#Loss Function and Optimizer

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-3
)

#Training
epochs = 300
training_losses = []

for epoch in range(epochs):

    model.train()

    #Forward pass
    predictions = model(X_train_tensor)

    #Calculate loss
    loss = criterion(
        predictions,
        y_train_tensor
    )

    
    optimizer.zero_grad()

    
    loss.backward() #backpropagation

   
    optimizer.step()  #Update weights

  
    training_losses.append(loss.item())   #Store loss

    
    if (epoch + 1) % 30 == 0: #Display progress

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.4f}"
        )



#Prediction


model.eval()

with torch.no_grad():

    y_pred_tensor = model(X_test_tensor)


#Convert PyTorch tensor to NumPy
y_pred_scaled = y_pred_tensor.numpy().flatten()

y_pred = np.expm1(y_pred_scaled * y_std + y_mean)



#evaluation
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