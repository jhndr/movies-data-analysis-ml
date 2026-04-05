import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.titlesize': 10,
                     'axes.titleweight': 'bold', 'axes.labelsize': 9,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'figure.dpi': 150})

# ── 1. Load & Prepare Data ───────────────────────────────────────────────────
df = pd.read_csv('cleaned_movies.csv')

# Drop rows with missing key features
df = df.dropna(subset=['YEAR', 'PRIMARY_GENRE', 'RATING'])
df = df[df['RunTime'] > 0]
df = df[df['VOTES'] >= 0]

# Feature engineering
df['LOG_VOTES'] = np.log1p(df['VOTES'])           # log-transform votes
df['LOG_GROSS'] = np.log1p(df['Gross_M'])          # log-transform gross

# Encode genre
le = LabelEncoder()
df['GENRE_ENC'] = le.fit_transform(df['PRIMARY_GENRE'])

features = ['YEAR', 'GENRE_ENC', 'LOG_VOTES', 'RunTime', 'LOG_GROSS']
target   = 'RATING'

X = df[features].values
y = df[target].values

print(f"Dataset size after preparation: {len(df)} records")
print(f"Features: {features}")
print(f"Target:   {target}  (mean={y.mean():.2f}, std={y.std():.2f})")

# ── 2. Train / Test Split ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

print(f"\nTrain samples: {len(X_train)}   Test samples: {len(X_test)}")

# ── 3. Model Training ─────────────────────────────────────────────────────────
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest':     RandomForestRegressor(n_estimators=100, max_depth=10,
                                               random_state=42, n_jobs=-1),
    'Neural Network\n(MLP)': MLPRegressor(hidden_layer_sizes=(128, 64, 32),
                                          activation='relu', solver='adam',
                                          max_iter=500, random_state=42,
                                          early_stopping=True, validation_fraction=0.1)
}

results = {}
predictions = {}

print("\n── Model Results ──────────────────────────────────────────────────────")
print(f"{'Model':<25} {'MAE':>7} {'RMSE':>7} {'R²':>7}")
print("-" * 50)

for name, model in models.items():
    label = name.replace('\n', ' ')
    if 'Linear' in name:
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
    elif 'Random' in name:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    results[label]     = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    predictions[label] = y_pred
    print(f"{label:<25} {mae:>7.4f} {rmse:>7.4f} {r2:>7.4f}")

# ── 4. Feature Importance (Random Forest) ────────────────────────────────────
rf_model      = models['Random Forest']
feature_names = ['Year', 'Genre', 'Log Votes', 'Runtime', 'Log Gross']
importances   = rf_model.feature_importances_
fi_idx        = np.argsort(importances)[::-1]

print("\n── Feature Importances (Random Forest) ────────────────────────────────")
for i in fi_idx:
    print(f"  {feature_names[i]:<12}: {importances[i]:.4f}")

# ── 5. Visualizations ─────────────────────────────────────────────────────────

# --- Chart A: Actual vs Predicted (best model = Random Forest) ---------------
best = 'Random Forest'
fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(y_test, predictions[best], alpha=0.3, s=8, color='steelblue')
mn, mx = y_test.min(), y_test.max()
ax.plot([mn, mx], [mn, mx], 'r--', linewidth=1.2, label='Perfect fit')
ax.set_xlabel('Actual Rating')
ax.set_ylabel('Predicted Rating')
ax.set_title(f'Actual vs Predicted Rating\n(Random Forest)')
ax.legend(fontsize=8)
r2_val = results[best]['R2']
ax.text(0.05, 0.92, f"R² = {r2_val:.4f}", transform=ax.transAxes,
        fontsize=8, color='darkred')
plt.tight_layout()
plt.savefig('ml_chart1_actual_vs_predicted.png', bbox_inches='tight')
plt.close()
print("\nSaved: ml_chart1_actual_vs_predicted.png")

# --- Chart B: Feature Importance ---------------------------------------------
fig, ax = plt.subplots(figsize=(5, 3.5))
colors = ['#2E75B6' if i == fi_idx[0] else '#7BAFD4' for i in range(len(feature_names))]
bars = ax.bar([feature_names[i] for i in fi_idx],
              [importances[i] for i in fi_idx], color=colors)
ax.set_ylabel('Importance Score')
ax.set_title('Feature Importance\n(Random Forest)')
for bar, val in zip(bars, [importances[i] for i in fi_idx]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.3f}', ha='center', fontsize=7)
plt.tight_layout()
plt.savefig('ml_chart2_feature_importance.png', bbox_inches='tight')
plt.close()
print("Saved: ml_chart2_feature_importance.png")

# --- Chart C: Model Comparison (MAE, RMSE, R²) --------------------------------
model_labels = list(results.keys())
mae_vals  = [results[m]['MAE']  for m in model_labels]
rmse_vals = [results[m]['RMSE'] for m in model_labels]
r2_vals   = [results[m]['R2']   for m in model_labels]
x = np.arange(len(model_labels))
w = 0.28

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(x - w, mae_vals,  w, label='MAE',  color='#2E75B6')
ax.bar(x,     rmse_vals, w, label='RMSE', color='#ED7D31')
ax.bar(x + w, r2_vals,   w, label='R²',   color='#70AD47')
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=8)
ax.set_ylabel('Score')
ax.set_title('Model Comparison: MAE, RMSE, R²')
ax.legend(fontsize=8)
for i, (mae, rmse, r2) in enumerate(zip(mae_vals, rmse_vals, r2_vals)):
    ax.text(i - w, mae  + 0.005, f'{mae:.3f}',  ha='center', fontsize=6.5)
    ax.text(i,     rmse + 0.005, f'{rmse:.3f}', ha='center', fontsize=6.5)
    ax.text(i + w, r2   + 0.005, f'{r2:.3f}',  ha='center', fontsize=6.5)
plt.tight_layout()
plt.savefig('ml_chart3_model_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: ml_chart3_model_comparison.png")

# --- Chart D: Residuals Distribution (Random Forest) -------------------------
residuals = y_test - predictions[best]
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.hist(residuals, bins=40, color='mediumslateblue', edgecolor='white', linewidth=0.4)
ax.axvline(0, color='red', linestyle='--', linewidth=1.2, label='Zero error')
ax.axvline(residuals.mean(), color='orange', linestyle='--', linewidth=1.2,
           label=f'Mean: {residuals.mean():.3f}')
ax.set_xlabel('Residual (Actual - Predicted)')
ax.set_ylabel('Count')
ax.set_title('Residuals Distribution\n(Random Forest)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('ml_chart4_residuals.png', bbox_inches='tight')
plt.close()
print("Saved: ml_chart4_residuals.png")

# --- Chart E: Rating distribution train vs test ------------------------------
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.hist(y_train, bins=30, alpha=0.6, color='steelblue', label='Train set', edgecolor='white')
ax.hist(y_test,  bins=30, alpha=0.6, color='coral',     label='Test set',  edgecolor='white')
ax.set_xlabel('IMDb Rating')
ax.set_ylabel('Count')
ax.set_title('Rating Distribution:\nTrain vs Test Split')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('ml_chart5_train_test_distribution.png', bbox_inches='tight')
plt.close()
print("Saved: ml_chart5_train_test_distribution.png")

print("\nAll charts saved. ML pipeline complete.")
print(f"\nBest model: Random Forest")
print(f"  MAE  = {results['Random Forest']['MAE']:.4f}")
print(f"  RMSE = {results['Random Forest']['RMSE']:.4f}")
print(f"  R²   = {results['Random Forest']['R2']:.4f}")
