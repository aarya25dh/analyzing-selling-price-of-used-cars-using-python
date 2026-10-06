import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/processed/car_new_details.csv")
X = df.drop(columns = ["Price","Model","Year","Owner"])
y = df["Price"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

