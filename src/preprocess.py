import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib
import os

def load_and_preprocess(filepath, sample_size=100000, random_state=42):
    """
    Full preprocessing pipeline for crop yield dataset.
    Returns: X (features), y (target), df_clean (full cleaned dataframe)
    """

    print("=" * 50)
    print("PHASE 4: DATA PREPROCESSING")
    print("=" * 50)

    # ── Step 1: Load data ──────────────────────────────
    print("\n[1/7] Loading data...")
    df = pd.read_csv(filepath)
    print(f"     Original shape: {df.shape}")

    # ── Step 2: Sample for performance ────────────────
    print(f"\n[2/7] Sampling {sample_size:,} rows...")
    df = df.sample(n=sample_size, random_state=random_state)
    df = df.reset_index(drop=True)
    print(f"     Sample shape: {df.shape}")

    # ── Step 3: Remove negative yields ────────────────
    print("\n[3/7] Removing invalid yields...")
    before = len(df)
    df = df[df['Yield_tons_per_hectare'] > 0]
    after  = len(df)
    print(f"     Removed {before - after} negative yield rows")
    print(f"     Clean shape: {df.shape}")

    # ── Step 4: Remove outliers using IQR method ──────
    print("\n[4/7] Removing outliers...")
    Q1  = df['Yield_tons_per_hectare'].quantile(0.25)
    Q3  = df['Yield_tons_per_hectare'].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    before = len(df)
    df = df[(df['Yield_tons_per_hectare'] >= lower) &
            (df['Yield_tons_per_hectare'] <= upper)]
    print(f"     Yield range kept: {lower:.2f} to {upper:.2f} tons/ha")
    print(f"     Removed {before - len(df)} outlier rows")

    # ── Step 5: Convert boolean columns ───────────────
    print("\n[5/7] Converting boolean columns...")
    df['Fertilizer_Used'] = df['Fertilizer_Used'].astype(int)
    df['Irrigation_Used'] = df['Irrigation_Used'].astype(int)
    print("     Fertilizer_Used & Irrigation_Used → 0/1 ✅")

    # ── Step 6: Encode categorical columns ────────────
    print("\n[6/7] Encoding categorical columns...")
    cat_cols     = ['Region', 'Soil_Type', 'Crop', 'Weather_Condition']
    encoders     = {}

    for col in cat_cols:
        le           = LabelEncoder()
        df[col]      = le.fit_transform(df[col])
        encoders[col] = le
        print(f"     {col}: {list(le.classes_)}")

    # ── Step 7: Save encoders & cleaned data ──────────
    print("\n[7/7] Saving processed data & encoders...")
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models',         exist_ok=True)

    df.to_csv('data/processed/crop_yield_clean.csv', index=False)
    joblib.dump(encoders, 'models/encoders.pkl')
    print("     data/processed/crop_yield_clean.csv ✅")
    print("     models/encoders.pkl ✅")

    # ── Final summary ──────────────────────────────────
    print("\n" + "=" * 50)
    print("PREPROCESSING COMPLETE")
    print("=" * 50)
    print(f"Final shape      : {df.shape}")
    print(f"Features         : {df.shape[1] - 1}")
    print(f"Target range     : {df['Yield_tons_per_hectare'].min():.2f}"
          f" to {df['Yield_tons_per_hectare'].max():.2f} tons/ha")
    print(f"Avg yield        : {df['Yield_tons_per_hectare'].mean():.2f} tons/ha")
    print("\nFinal column dtypes:")
    print(df.dtypes)

    # ── Prepare X and y ───────────────────────────────
    X = df.drop('Yield_tons_per_hectare', axis=1)
    y = df['Yield_tons_per_hectare']

    return X, y, df, encoders


# ── Run directly ───────────────────────────────────────
if __name__ == "__main__":
    X, y, df_clean, encoders = load_and_preprocess(
        'data/raw/crop_yield.csv',
        sample_size=100000
    )

    print(f"\nX shape : {X.shape}")
    print(f"y shape : {y.shape}")
    print(f"\nFirst 5 rows of X:")
    print(X.head())
    print(f"\nFirst 5 values of y:")
    print(y.head())