import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

print("=======================================================")
print("FERTILIZER MODEL TRAINING")
print("=======================================================")

# [1] Load data
print("[1/6] Loading fertilizer data...")
df = pd.read_csv("data/raw/fertilizer_recommendation.csv")
print(f"      Shape: {df.shape}")
print(f"      Columns: {list(df.columns)}")

# [2] Clean column names (strip spaces)
df.columns = df.columns.str.strip()
print("[2/6] Cleaned column names...")

# [3] Encode categorical columns
print("[3/6] Encoding categorical features...")
encoders = {}
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Separate target from features
# Common target column names — adjust if yours is different
target_col = None
for candidate in ["Fertilizer Name", "Fertilizer", "fertilizer", "label", "Label"]:
    if candidate in df.columns:
        target_col = candidate
        break

if target_col is None:
    # Use last column as target
    target_col = df.columns[-1]

print(f"      Target column: '{target_col}'")
categorical_cols = [c for c in categorical_cols if c != target_col]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"      Encoded: {col}")

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(df[target_col].astype(str))
encoders["__target__"] = le_target
print(f"      Target classes: {list(le_target.classes_)}")

# [4] Split features and target
print("[4/6] Splitting features and target...")
X = df.drop(columns=[target_col])
feature_names = list(X.columns)
print(f"      Features: {feature_names}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"      Training rows : {len(X_train):,}")
print(f"      Testing rows  : {len(X_test):,}")

# [5] Train model
print("[5/6] Training Random Forest Classifier...")
print("      This may take a moment...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("      ✅ Training done!")

# [6] Evaluate
print("[6/6] Evaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print()
print("=======================================================")
print("MODEL PERFORMANCE")
print("=======================================================")
print(f"  Accuracy : {acc*100:.2f}%")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_target.classes_))

# Save models
os.makedirs("models", exist_ok=True)
with open("models/fertilizer_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/fertilizer_encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
with open("models/fertilizer_features.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("=======================================================")
print("FILES SAVED")
print("=======================================================")
print("  models/fertilizer_model.pkl     ✅")
print("  models/fertilizer_encoders.pkl  ✅")
print("  models/fertilizer_features.pkl  ✅")