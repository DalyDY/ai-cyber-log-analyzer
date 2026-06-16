import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

df = pd.read_csv(
    "data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
)

df.columns = df.columns.str.strip()

print("Before cleaning:", df.shape)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

df["Target"] = df["Label"].map({
    "BENIGN": 0,
    "DDoS": 1
})

print("After cleaning:", df.shape)

print("\nLabel counts:")
print(df["Label"].value_counts())

print("\nTarget counts:")
print(df["Target"].value_counts())

print("\nInfinite values:")
print(np.isinf(df.select_dtypes(include=[np.number])).sum().sum())

# Train model

X = df.drop(columns=["Label", "Target"])
y = df["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Logistic Regression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, pred))

print("\nLogistic Regression Report:")
print(classification_report(y_test, pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

joblib.dump(model, "model.pkl")

print("Model saved.")