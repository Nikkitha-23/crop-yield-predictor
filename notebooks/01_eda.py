import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 120

# ── Load & Sample ──────────────────────────────────────
print("Loading data...")
df_full = pd.read_csv('data/raw/crop_yield.csv')
df = df_full.sample(n=50000, random_state=42)  # 50k sample
print(f"Full dataset : {df_full.shape}")
print(f"Working sample: {df.shape}")
print(f"\nColumns: {list(df.columns)}")

# ── Quick Stats ────────────────────────────────────────
print("\n" + "="*50)
print("BASIC STATISTICS")
print("="*50)
print(df.describe())

# ── 1. Yield Distribution ──────────────────────────────
print("\nGenerating Plot 1...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('1. Yield Distribution', fontsize=14, fontweight='bold')

axes[0].hist(df['Yield_tons_per_hectare'], bins=50,
             color='steelblue', edgecolor='white', alpha=0.85)
axes[0].set_title('Histogram of Yield')
axes[0].set_xlabel('Yield (tons/hectare)')
axes[0].set_ylabel('Count')

axes[1].boxplot(df['Yield_tons_per_hectare'],
                patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.7))
axes[1].set_title('Boxplot of Yield')
axes[1].set_ylabel('Yield (tons/hectare)')
plt.tight_layout()
plt.savefig('notebooks/01_yield_distribution.png')
plt.show()
print("✅ Plot 1 done")

# ── 2. Yield by Crop ───────────────────────────────────
print("Generating Plot 2...")
fig, ax = plt.subplots(figsize=(14, 6))
crop_yield = (df.groupby('Crop')['Yield_tons_per_hectare']
                .mean().sort_values(ascending=False))
bars = ax.bar(crop_yield.index, crop_yield.values,
              color=sns.color_palette("muted", len(crop_yield)),
              edgecolor='white')
ax.set_title('2. Average Yield by Crop', fontsize=14, fontweight='bold')
ax.set_xlabel('Crop')
ax.set_ylabel('Average Yield (tons/hectare)')
ax.tick_params(axis='x', rotation=30)
for bar, val in zip(bars, crop_yield.values):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.05,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig('notebooks/02_yield_by_crop.png')
plt.show()
print("✅ Plot 2 done")

# ── 3. Yield by Region ─────────────────────────────────
print("Generating Plot 3...")
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x='Region',
            y='Yield_tons_per_hectare',
            palette='muted', ax=ax)
ax.set_title('3. Yield by Region', fontsize=14, fontweight='bold')
ax.set_xlabel('Region')
ax.set_ylabel('Yield (tons/hectare)')
plt.tight_layout()
plt.savefig('notebooks/03_yield_by_region.png')
plt.show()
print("✅ Plot 3 done")

# ── 4. Numerical Features vs Yield ────────────────────
print("Generating Plot 4...")
num_features = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('4. Features vs Yield', fontsize=14, fontweight='bold')
sample = df.sample(3000, random_state=42)
for ax, feat in zip(axes, num_features):
    ax.scatter(sample[feat],
               sample['Yield_tons_per_hectare'],
               alpha=0.3, s=10, color='steelblue')
    ax.set_xlabel(feat)
    ax.set_ylabel('Yield (tons/hectare)')
    ax.set_title(f'{feat} vs Yield')
plt.tight_layout()
plt.savefig('notebooks/04_numerical_vs_yield.png')
plt.show()
print("✅ Plot 4 done")

# ── 5. Fertilizer & Irrigation ────────────────────────
print("Generating Plot 5...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('5. Fertilizer & Irrigation Impact',
             fontsize=14, fontweight='bold')
for ax, col in zip(axes, ['Fertilizer_Used', 'Irrigation_Used']):
    sns.boxplot(data=df, x=col,
                y='Yield_tons_per_hectare',
                palette='muted', ax=ax)
    ax.set_title(f'{col} vs Yield')
    ax.set_xlabel(col)
    ax.set_ylabel('Yield (tons/hectare)')
plt.tight_layout()
plt.savefig('notebooks/05_fertilizer_irrigation.png')
plt.show()
print("✅ Plot 5 done")

# ── 6. Weather Condition ──────────────────────────────
print("Generating Plot 6...")
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x='Weather_Condition',
            y='Yield_tons_per_hectare',
            palette='muted', ax=ax)
ax.set_title('6. Yield by Weather Condition',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Weather Condition')
ax.set_ylabel('Yield (tons/hectare)')
plt.tight_layout()
plt.savefig('notebooks/06_weather_condition.png')
plt.show()
print("✅ Plot 6 done")

# ── 7. Correlation Heatmap ────────────────────────────
print("Generating Plot 7...")
fig, ax = plt.subplots(figsize=(8, 6))
num_df = df[['Rainfall_mm', 'Temperature_Celsius',
             'Days_to_Harvest', 'Yield_tons_per_hectare']]
corr = num_df.corr()
sns.heatmap(corr, annot=True, fmt='.2f',
            cmap='coolwarm', center=0,
            square=True, linewidths=0.5, ax=ax)
ax.set_title('7. Correlation Heatmap',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/07_correlation_heatmap.png')
plt.show()
print("✅ Plot 7 done")

# ── Key Insights ──────────────────────────────────────
print("\n" + "="*50)
print("KEY EDA INSIGHTS")
print("="*50)
print(f"Average yield     : {df['Yield_tons_per_hectare'].mean():.2f} tons/ha")
print(f"Max yield         : {df['Yield_tons_per_hectare'].max():.2f} tons/ha")
print(f"Min yield         : {df['Yield_tons_per_hectare'].min():.2f} tons/ha")
print(f"Negative yields   : {(df['Yield_tons_per_hectare'] < 0).sum()} rows ⚠️")
print(f"Fertilizer used % : {df['Fertilizer_Used'].mean()*100:.1f}%")
print(f"Irrigation used % : {df['Irrigation_Used'].mean()*100:.1f}%")
print(f"Unique crops      : {df['Crop'].nunique()}")
print(f"Unique regions    : {df['Region'].nunique()}")
print(f"Unique soil types : {df['Soil_Type'].nunique()}")
print(f"Weather types     : {df['Weather_Condition'].nunique()}")
print("\n✅ EDA Complete! Plots saved in notebooks/ folder")