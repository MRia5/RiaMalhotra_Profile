"""
============================================================
  Fraud Detection Engine
  XGBoost + Isolation Forest Ensemble Pipeline
============================================================
  Handles:
    • Severe class imbalance via SMOTE
    • Custom threshold optimisation (F1 / precision-recall)
    • Feature engineering (transaction velocity, time features)
    • Ensemble: supervised XGBoost + unsupervised Isolation Forest
    • Evaluation: ROC-AUC, precision-recall, confusion matrix
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection    import train_test_split, StratifiedKFold
from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.ensemble           import IsolationForest
from sklearn.metrics            import (roc_auc_score, classification_report,
                                        confusion_matrix, precision_recall_curve,
                                        roc_curve, f1_score)
from xgboost                    import XGBClassifier
from imblearn.over_sampling     import SMOTE


# ─────────────────────────────────────────────────────────
#  GENERATE SYNTHETIC DATASET
#  Mimics credit card transaction data with realistic imbalance
# ─────────────────────────────────────────────────────────

np.random.seed(42)

def generate_transactions(n_total=50_000, fraud_rate=0.015):
    """Generate a synthetic transaction dataset."""
    n_fraud  = int(n_total * fraud_rate)
    n_legit  = n_total - n_fraud

    # Legitimate transactions
    legit = pd.DataFrame({
        'amount':            np.random.lognormal(mean=4.0, sigma=1.2, size=n_legit),
        'hour':              np.random.choice(range(8, 23), size=n_legit),       # business hours
        'day_of_week':       np.random.choice(range(0, 7),  size=n_legit),
        'merchant_category': np.random.choice(['retail','food','travel','online','atm'],
                                               size=n_legit, p=[0.35,0.25,0.15,0.20,0.05]),
        'n_txn_last_1h':     np.random.poisson(lam=1.0, size=n_legit),
        'n_txn_last_24h':    np.random.poisson(lam=6.0, size=n_legit),
        'distance_km':       np.abs(np.random.normal(loc=5,   scale=10,  size=n_legit)),
        'is_foreign':        np.random.choice([0, 1], size=n_legit, p=[0.95, 0.05]),
        'account_age_days':  np.random.randint(180, 3650, size=n_legit),
        'is_fraud':          0,
    })

    # Fraudulent transactions (different distribution)
    fraud = pd.DataFrame({
        'amount':            np.random.lognormal(mean=5.5, sigma=1.5, size=n_fraud),
        'hour':              np.random.choice(range(0, 6), size=n_fraud),        # odd hours
        'day_of_week':       np.random.choice(range(0, 7), size=n_fraud),
        'merchant_category': np.random.choice(['online','atm','travel','retail','food'],
                                               size=n_fraud, p=[0.40,0.30,0.15,0.10,0.05]),
        'n_txn_last_1h':     np.random.poisson(lam=4.0, size=n_fraud),           # burst pattern
        'n_txn_last_24h':    np.random.poisson(lam=12.0, size=n_fraud),
        'distance_km':       np.abs(np.random.normal(loc=200, scale=150, size=n_fraud)),
        'is_foreign':        np.random.choice([0, 1], size=n_fraud, p=[0.50, 0.50]),
        'account_age_days':  np.random.randint(1, 90, size=n_fraud),             # new accounts
        'is_fraud':          1,
    })

    df = pd.concat([legit, fraud]).sample(frac=1, random_state=42).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────

def engineer_features(df):
    """Create derived features from raw transaction data."""
    df = df.copy()

    # Log-transform amount (heavy right tail)
    df['log_amount'] = np.log1p(df['amount'])

    # Is odd-hour transaction?
    df['is_odd_hour'] = df['hour'].apply(lambda h: 1 if h < 6 or h > 22 else 0)

    # Velocity ratio
    df['velocity_ratio'] = df['n_txn_last_1h'] / (df['n_txn_last_24h'] + 1)

    # High distance flag
    df['high_distance'] = (df['distance_km'] > 100).astype(int)

    # New account flag
    df['new_account'] = (df['account_age_days'] < 30).astype(int)

    # Encode merchant category
    le = LabelEncoder()
    df['merchant_enc'] = le.fit_transform(df['merchant_category'])

    return df


# ─────────────────────────────────────────────────────────
#  LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────

print("=" * 55)
print("  Fraud Detection Engine")
print("=" * 55)

print("\n[1/6] Generating synthetic transaction dataset...")
df_raw = generate_transactions(n_total=50_000)
print(f"      Total transactions : {len(df_raw):,}")
print(f"      Fraud rate         : {df_raw['is_fraud'].mean()*100:.2f}%")

print("\n[2/6] Engineering features...")
df = engineer_features(df_raw)

FEATURES = [
    'log_amount', 'hour', 'day_of_week', 'merchant_enc',
    'n_txn_last_1h', 'n_txn_last_24h', 'distance_km',
    'is_foreign', 'account_age_days', 'is_odd_hour',
    'velocity_ratio', 'high_distance', 'new_account',
]

X = df[FEATURES].values
y = df['is_fraud'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)


# ─────────────────────────────────────────────────────────
#  SMOTE — Oversample minority class
# ─────────────────────────────────────────────────────────

print("\n[3/6] Applying SMOTE to balance training data...")
smote = SMOTE(sampling_strategy=0.2, random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
print(f"      Before SMOTE: {np.bincount(y_train)}")
print(f"      After  SMOTE: {np.bincount(y_res)}")


# ─────────────────────────────────────────────────────────
#  MODEL 1 — XGBoost (supervised)
# ─────────────────────────────────────────────────────────

print("\n[4/6] Training XGBoost classifier...")
xgb = XGBClassifier(
    n_estimators     = 300,
    max_depth        = 6,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    use_label_encoder= False,
    eval_metric      = 'logloss',
    random_state     = 42,
    n_jobs           = -1,
)
xgb.fit(X_res, y_res, eval_set=[(X_test, y_test)], verbose=False)
xgb_probs = xgb.predict_proba(X_test)[:, 1]


# ─────────────────────────────────────────────────────────
#  MODEL 2 — Isolation Forest (unsupervised)
# ─────────────────────────────────────────────────────────

print("[5/6] Fitting Isolation Forest anomaly detector...")
iso = IsolationForest(
    n_estimators  = 200,
    contamination = 0.015,
    random_state  = 42,
    n_jobs        = -1,
)
iso.fit(X_train)
# Convert anomaly scores to [0,1] range (higher = more anomalous)
iso_scores = -iso.score_samples(X_test)
iso_probs  = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min())


# ─────────────────────────────────────────────────────────
#  ENSEMBLE — Weighted combination
# ─────────────────────────────────────────────────────────

W_XGB = 0.75
W_ISO = 0.25
ensemble_probs = W_XGB * xgb_probs + W_ISO * iso_probs


# ─────────────────────────────────────────────────────────
#  THRESHOLD OPTIMISATION — Maximise F1
# ─────────────────────────────────────────────────────────

precisions, recalls, thresholds = precision_recall_curve(y_test, ensemble_probs)
f1_scores  = 2 * precisions * recalls / (precisions + recalls + 1e-9)
best_idx   = np.argmax(f1_scores[:-1])
best_thresh = thresholds[best_idx]

y_pred = (ensemble_probs >= best_thresh).astype(int)

print(f"\n[6/6] Best threshold (F1-optimal): {best_thresh:.3f}")


# ─────────────────────────────────────────────────────────
#  EVALUATION
# ─────────────────────────────────────────────────────────

roc_auc = roc_auc_score(y_test, ensemble_probs)
f1      = f1_score(y_test, y_pred)
cm      = confusion_matrix(y_test, y_pred)

print("\n" + "=" * 55)
print("  RESULTS")
print("=" * 55)
print(f"  ROC-AUC : {roc_auc:.4f}")
print(f"  F1      : {f1:.4f}")
print(f"\n  Confusion Matrix:\n{cm}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Legit','Fraud'])}")


# ─────────────────────────────────────────────────────────
#  CHARTS
# ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Fraud Detection Engine — Evaluation", fontweight='bold')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, ensemble_probs)
axes[0].plot(fpr, tpr, color='#1a7a3c', lw=2, label=f'ROC-AUC = {roc_auc:.3f}')
axes[0].plot([0,1],[0,1], 'k--', lw=0.8)
axes[0].set_title("ROC Curve")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

# Precision-Recall Curve
axes[1].plot(recalls[:-1], precisions[:-1], color='#4c72b0', lw=2)
axes[1].axvline(recalls[best_idx], color='red', linestyle='--', label=f'Threshold={best_thresh:.2f}')
axes[1].set_title("Precision-Recall Curve")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()

# Feature Importance (XGBoost)
importances = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=True)
axes[2].barh(importances.index, importances.values, color='#dd8452')
axes[2].set_title("Feature Importance (XGBoost)")
axes[2].set_xlabel("Importance Score")

plt.tight_layout()
plt.savefig('fraud_detection_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n📊 Chart saved as fraud_detection_results.png")
print("\n  ⚠️ Trained on synthetic data for demonstration purposes only.")
