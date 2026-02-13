import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("/home/surya/programs/python-learning/ML_lab/StudentsPerformance.csv")

df["final_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)

label_encoder = LabelEncoder()

df["parental level of education"] = label_encoder.fit_transform(
    df["parental level of education"]
)

df["test preparation course"] = label_encoder.fit_transform(
    df["test preparation course"]
)

X = df[
    [
        "parental level of education",
        "test preparation course"
    ]
]

np.random.seed(42)
df["study_hours"] = np.random.uniform(1, 6, len(df))
df["attendance"] = np.random.uniform(60, 100, len(df))
df["sleep_hours"] = np.random.uniform(4, 9, len(df))

X["study_hours"] = df["study_hours"]
X["attendance"] = df["attendance"]
X["sleep_hours"] = df["sleep_hours"]

y = df["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)
