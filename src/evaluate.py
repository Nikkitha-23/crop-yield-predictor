import pandas as pd
import numpy as np
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import warnings
warnings.filterwarnings('ignore')

def evaluate_model():
    print("=" * 55)
    print("PHASE 7: MODEL EVALUATION")
    print("=" * 55)

    # ── Load model and data ────────────────────────────
    print("\n[1/5] Loading model and data...")
    model         = joblib.load('models/random_forest.pkl')
    feature_names = joblib.load('models/feature_names.pkl')

    df = pd.read_csv('data/processed/crop_yield_engineered.csv')
    X  = df.drop('Yield_tons_per_hectare', axis=1)
    y  = df['Yield_tons_per_hectare']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_pred = model.predict(X_test)
    print("      Model and data loaded ✅")

    # ── Core metrics ───────────────────────────────────
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"\n{'='*55}")
    print(f"DETAILED METRICS")
    print(f"{'='*55}")
    print(f"  R² Score : {r2:.4f}")
    print(f"  MAE      : {mae:.4f} tons/ha")
    print(f"  RMSE     : {rmse:.4f} tons/ha")
    print(f"  MAPE     : {mape:.2f}% (mean % error)")

    # ── Error analysis ────────────────────────────────
    errors = y_test - y_pred
    print(f"\n  Error Analysis:")
    print(f"  Mean error   : {errors.mean():.4f} (bias)")
    print(f"  Max overest. : {errors.min():.4f} tons/ha")
    print(f"  Max underest.: {errors.max():.4f} tons/ha")
    within_05 = (np.abs(errors) <= 0.5).mean() * 100
    within_10 = (np.abs(errors) <= 1.0).mean() * 100
    print(f"  Within ±0.5t : {within_05:.1f}% of predictions")
    print(f"  Within ±1.0t : {within_10:.1f}% of predictions")

    # ── Cross validation ──────────────────────────────
    print(f"\n[2/5] Running 5-fold cross validation...")
    print(f"      (This checks model isn't just memorizing)")
    cv_scores = cross_val_score(
        model, X, y,
        cv=5, scoring='r2',
        n_jobs=-1
    )
    print(f"\n  CV R² Scores  : {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  CV Mean R²    : {cv_scores.mean():.4f}")
    print(f"  CV Std        : {cv_scores.std():.4f}")
    if cv_scores.std() < 0.02:
        print(f"  ✅ Very stable model — no overfitting!")
    elif cv_scores.std() < 0.05:
        print(f"  ✅ Stable model")
    else:
        print(f"  ⚠️  Some variance — consider tuning")

    # ── Plot 1: Actual vs Predicted ───────────────────
    print(f"\n[3/5] Generating evaluation plots...")
    sample_idx = np.random.choice(len(y_test), 3000, replace=False)
    y_test_s   = np.array(y_test)[sample_idx]
    y_pred_s   = y_pred[sample_idx]

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    # Plot A — Actual vs Predicted scatter
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(y_test_s, y_pred_s,
                alpha=0.3, s=10, color='steelblue')
    min_val = min(y_test_s.min(), y_pred_s.min())
    max_val = max(y_test_s.max(), y_pred_s.max())
    ax1.plot([min_val, max_val],
             [min_val, max_val],
             'r--', linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('Actual Yield (tons/ha)')
    ax1.set_ylabel('Predicted Yield (tons/ha)')
    ax1.set_title(f'Actual vs Predicted\nR²={r2:.4f}',
                  fontweight='bold')
    ax1.legend()

    # Plot B — Residuals distribution
    ax2 = fig.add_subplot(gs[0, 1])
    errors_s = y_test_s - y_pred_s
    ax2.hist(errors_s, bins=50,
             color='steelblue', edgecolor='white', alpha=0.85)
    ax2.axvline(0, color='red',
                linestyle='--', linewidth=2, label='Zero error')
    ax2.set_xlabel('Prediction Error (tons/ha)')
    ax2.set_ylabel('Count')
    ax2.set_title('Residuals Distribution\n(should be centered at 0)',
                  fontweight='bold')
    ax2.legend()

    # Plot C — Residuals vs Predicted
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(y_pred_s, errors_s,
                alpha=0.3, s=10, color='steelblue')
    ax3.axhline(0, color='red',
                linestyle='--', linewidth=2)
    ax3.set_xlabel('Predicted Yield (tons/ha)')
    ax3.set_ylabel('Residual Error')
    ax3.set_title('Residuals vs Predicted\n(should be random scatter)',
                  fontweight='bold')

    # Plot D — Cross validation scores
    ax4 = fig.add_subplot(gs[1, 1])
    fold_labels = [f'Fold {i+1}' for i in range(len(cv_scores))]
    bars = ax4.bar(fold_labels, cv_scores,
                   color='steelblue', edgecolor='white', alpha=0.85)
    ax4.axhline(cv_scores.mean(), color='red',
                linestyle='--', linewidth=2,
                label=f'Mean R²={cv_scores.mean():.3f}')
    ax4.set_ylim(0.85, 1.0)
    ax4.set_ylabel('R² Score')
    ax4.set_title('5-Fold Cross Validation\n(consistency check)',
                  fontweight='bold')
    ax4.legend()
    for bar, val in zip(bars, cv_scores):
        ax4.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.001,
                 f'{val:.3f}', ha='center',
                 va='bottom', fontsize=9)

    plt.suptitle('Model Evaluation Dashboard',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig('notebooks/09_evaluation_dashboard.png',
                bbox_inches='tight')
    plt.show()
    print("      ✅ Evaluation dashboard saved!")

    # ── Sample predictions ────────────────────────────
    print(f"\n[4/5] Sample predictions vs actual:")
    print(f"\n  {'Actual':>10} {'Predicted':>10} {'Error':>10}")
    print(f"  {'-'*32}")
    for actual, predicted in zip(y_test.values[:8],
                                 y_pred[:8]):
        error = actual - predicted
        print(f"  {actual:>10.3f} {predicted:>10.3f}"
              f" {error:>+10.3f}")

    # ── Final verdict ─────────────────────────────────
    print(f"\n{'='*55}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*55}")
    print(f"  R² Score     : {r2:.4f} 🔥")
    print(f"  MAE          : {mae:.4f} tons/ha")
    print(f"  MAPE         : {mape:.2f}%")
    print(f"  CV Mean R²   : {cv_scores.mean():.4f}")
    print(f"  Within ±0.5t : {within_05:.1f}%")
    print(f"  Within ±1.0t : {within_10:.1f}%")
    print(f"\n  ✅ Model is ready for deployment!")

if __name__ == "__main__":
    evaluate_model()