import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error, r2_score)
import matplotlib.pyplot as plt
import joblib
import os
import time

def train_model():
    print("=" * 55)
    print("PHASE 6: TRAINING RANDOM FOREST MODEL")
    print("=" * 55)

    # ── Step 1: Load engineered data ──────────────────
    print("\n[1/6] Loading engineered data...")
    df = pd.read_csv('data/processed/crop_yield_engineered.csv')
    print(f"      Shape: {df.shape}")

    # ── Step 2: Split features and target ─────────────
    print("\n[2/6] Splitting features and target...")
    X = df.drop('Yield_tons_per_hectare', axis=1)
    y = df['Yield_tons_per_hectare']
    print(f"      X shape: {X.shape}")
    print(f"      y shape: {y.shape}")
    print(f"      Features: {list(X.columns)}")

    # ── Step 3: Train/Test split ───────────────────────
    # 80% training, 20% testing — standard split
    print("\n[3/6] Splitting train/test (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"      Training rows : {X_train.shape[0]:,}")
    print(f"      Testing rows  : {X_test.shape[0]:,}")

    # ── Step 4: Train Random Forest ───────────────────
    print("\n[4/6] Training Random Forest...")
    print("      This may take 1-2 minutes...")
    print("      Parameters:")
    print("        n_estimators = 100  (100 trees)")
    print("        max_depth    = 15   (tree depth limit)")
    print("        random_state = 42   (reproducibility)")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,           # use all CPU cores
        random_state=42,
        verbose=1
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start_time
    print(f"\n      ✅ Training done in {elapsed:.1f} seconds!")

    # ── Step 5: Evaluate on test set ──────────────────
    print("\n[5/6] Evaluating model...")
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"\n{'='*55}")
    print(f"MODEL PERFORMANCE ON TEST SET")
    print(f"{'='*55}")
    print(f"  R² Score  : {r2:.4f}  (closer to 1.0 = better)")
    print(f"  MAE       : {mae:.4f} tons/ha (avg prediction error)")
    print(f"  RMSE      : {rmse:.4f} tons/ha (penalizes big errors)")
    print(f"\n  Interpretation:")
    if r2 >= 0.85:
        print(f"  🔥 Excellent model! R²={r2:.2f}")
    elif r2 >= 0.70:
        print(f"  ✅ Good model! R²={r2:.2f}")
    elif r2 >= 0.50:
        print(f"  ⚠️  Moderate model. R²={r2:.2f}")
    else:
        print(f"  ❌ Needs improvement. R²={r2:.2f}")

    # ── Step 6: Feature Importance Plot ───────────────
    print("\n[6/6] Plotting feature importance...")
    importances = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#2196F3' if i >= len(importances)-3
              else '#90CAF9'
              for i in range(len(importances))]
    bars = ax.barh(importances.index,
                   importances.values,
                   color=colors, edgecolor='white')
    ax.set_title('Feature Importance — Random Forest',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score')

    for bar, val in zip(bars, importances.values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('notebooks/08_feature_importance.png')
    plt.show()

    # ── Save model & metadata ─────────────────────────
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest.pkl')

    # Save feature names for prediction later
    feature_names = list(X.columns)
    joblib.dump(feature_names, 'models/feature_names.pkl')

    print(f"\n{'='*55}")
    print(f"FILES SAVED")
    print(f"{'='*55}")
    print(f"  models/random_forest.pkl  ✅")
    print(f"  models/feature_names.pkl  ✅")
    print(f"  notebooks/08_feature_importance.png  ✅")

    print(f"\n{'='*55}")
    print(f"TOP 3 MOST IMPORTANT FEATURES")
    print(f"{'='*55}")
    top3 = importances.sort_values(ascending=False).head(3)
    for i, (feat, score) in enumerate(top3.items(), 1):
        print(f"  {i}. {feat}: {score:.4f}")

    return model, X_test, y_test, y_pred


if __name__ == "__main__":
    model, X_test, y_test, y_pred = train_model()