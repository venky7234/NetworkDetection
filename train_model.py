import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("data/nsl_kdd.csv")

# Encode all string columns
for col in ['protocol_type', 'service', 'flag', 'label']:
    df[col] = LabelEncoder().fit_transform(df[col])

# Only use 6 features
columns_to_use = ["duration", "protocol_type", "service", "src_bytes", "dst_bytes", "flag"]
X = df[columns_to_use]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open("backend/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as backend/model.pkl")
