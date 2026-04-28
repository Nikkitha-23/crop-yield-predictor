import pandas as pd
import numpy as np
import joblib
import os

def engineer_features(df):
    """
    Creates new features from existing columns.
    Input : cleaned dataframe
    Output: dataframe with extra feature columns
    """
    print("=" * 50)
    print("PHASE 5: FEATURE ENGINEERING")
    print("=" * 50)

    df = df.copy()
    original_cols = df.shape[1]

    # ── Feature 1: Rainfall-Temperature Ratio ─────────
    # High rainfall + low temp = very different from
    # high rainfall + high temp
    print("\n[1/5] Creating rainfall_temp_ratio...")
    df['rainfall_temp_ratio'] = (
        df['Rainfall_mm'] / (df['Temperature_Celsius'] + 1)
    )
    print("      rainfall_mm ÷ temperature ✅")

    # ── Feature 2: Farming Practice Score ─────────────
    # Combines fertilizer + irrigation into one score
    # 0 = neither, 1 = one of them, 2 = both
    print("\n[2/5] Creating farming_score...")
    df['farming_score'] = (
        df['Fertilizer_Used'] + df['Irrigation_Used']
    )
    print("      fertilizer + irrigation score (0/1/2) ✅")

    # ── Feature 3: Combined Farming Practice ──────────
    # Interaction feature — did farmer use BOTH together?
    print("\n[3/5] Creating fertilizer_irrigation interaction...")
    df['fert_irrig_combined'] = (
        df['Fertilizer_Used'] * df['Irrigation_Used']
    )
    print("      fertilizer × irrigation (0 or 1) ✅")

    # ── Feature 4: Rainfall Category ──────────────────
    # Bins continuous rainfall into 3 meaningful groups
    print("\n[4/5] Creating rainfall_category...")
    df['rainfall_category'] = pd.cut(
        df['Rainfall_mm'],
        bins  =[0, 400, 700, 1100],
        labels=[0, 1, 2]       # 0=Low, 1=Medium, 2=High
    ).astype(int)
    print("      Low(0-400mm) Medium(400-700mm) High(700+mm) ✅")

    # ── Feature 5: Temperature Category ───────────────
    print("\n[5/5] Creating temp_category...")
    df['temp_category'] = pd.cut(
        df['Temperature_Celsius'],
        bins  =[0, 20, 30, 50],
        labels=[0, 1, 2]       # 0=Cool, 1=Warm, 2=Hot
    ).astype(int)
    print("      Cool(0-20°C) Warm(20-30°C) Hot(30°C+) ✅")

    # ── Summary ───────────────────────────────────────
    new_cols = df.shape[1] - original_cols
    print(f"\n{'='*50}")
    print(f"FEATURE ENGINEERING COMPLETE")
    print(f"{'='*50}")
    print(f"Original features : {original_cols}")
    print(f"New features added: {new_cols}")
    print(f"Total features    : {df.shape[1]}")
    print(f"\nAll columns:")
    for col in df.columns:
        print(f"  → {col}")

    return df


if __name__ == "__main__":
    # Load cleaned data
    print("Loading preprocessed data...")
    df = pd.read_csv('data/processed/crop_yield_clean.csv')
    print(f"Loaded shape: {df.shape}")

    # Engineer features
    df_engineered = engineer_features(df)

    # Save
    os.makedirs('data/processed', exist_ok=True)
    df_engineered.to_csv(
        'data/processed/crop_yield_engineered.csv',
        index=False
    )

    print(f"\nSaved → data/processed/crop_yield_engineered.csv")
    print(f"Final shape: {df_engineered.shape}")

    # Quick stats on new features
    print("\nNew feature statistics:")
    new_features = [
        'rainfall_temp_ratio', 'farming_score',
        'fert_irrig_combined', 'rainfall_category',
        'temp_category'
    ]
    print(df_engineered[new_features].describe().round(2))