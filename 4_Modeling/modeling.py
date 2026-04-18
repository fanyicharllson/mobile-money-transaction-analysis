#!/usr/bin/env python3
"""
# STATISTICAL MODELING & PREDICTION NOTEBOOK
# CSC 3221 - Introduction to Data Science
# ICT University Cameroon
# Target: User Activity Classification (Low / Medium / High)
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report, roc_auc_score,
                              roc_curve, ConfusionMatrixDisplay)
from sklearn.inspection import permutation_importance
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

os.makedirs('/home/claude/project/4_Modeling/results', exist_ok=True)

COLORS = ['#028090', '#F0A500', '#E63946', '#02C39A', '#8B5CF6']
plt.rcParams.update({'figure.dpi': 120, 'axes.spines.top': False, 'axes.spines.right': False})

print("=" * 60)
print("STEP 1: LOAD & PREPARE DATA")
print("=" * 60)

df = pd.read_csv('/home/claude/project/2_Data_Cleaning/cleaned_data.csv')

# Feature selection
FEATURE_COLS = [
    'age', 'gender_encoded', 'zone_encoded', 'education_encoded', 'income_encoded',
    'household_size', 'smartphone_encoded', 'years_using_mm',
    'total_transactions', 'avg_transaction_amount', 'tx_per_month',
    'send_receive_ratio', 'weekend_ratio', 'days_since_last_tx',
    'avg_monthly_amount', 'tx_type_diversity', 'total_fees_paid',
    'std_transaction_amount', 'max_transaction_amount'
]
TARGET = 'activity_label'

# Handle boolean columns from get_dummies
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)

# Only keep columns that exist
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]
df_model = df[FEATURE_COLS + [TARGET, 'activity_segment']].dropna()

X = df_model[FEATURE_COLS].values
y = df_model[TARGET].values
y_labels = df_model['activity_segment'].values

print(f"Features used: {len(FEATURE_COLS)}")
print(f"Samples: {len(X)}")
print(f"Class distribution: {pd.Series(y_labels).value_counts().to_dict()}")

# Train/Validation/Test split (70/15/15)
X_train, X_temp, y_train, y_temp, yl_train, yl_temp = train_test_split(
    X, y, y_labels, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test, yl_val, yl_test = train_test_split(
    X_temp, y_temp, yl_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"\nTrain: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)

print("=" * 60)
print("STEP 2: BASELINE MODEL")
print("=" * 60)

most_frequent_class = np.bincount(y_train).argmax()
baseline_preds = np.full(len(y_test), most_frequent_class)
baseline_acc = accuracy_score(y_test, baseline_preds)
print(f"Baseline (most frequent class = {most_frequent_class}): {baseline_acc:.3f} accuracy")

print("=" * 60)
print("STEP 3: MODEL TRAINING")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=6, min_samples_split=4, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_split=3,
                                              random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,
                                                       max_depth=4, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance')
}

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    if name in ['Logistic Regression', 'KNN']:
        X_tr, X_v = X_train_sc, X_val_sc
    else:
        X_tr, X_v = X_train, X_val

    model.fit(X_tr, y_train)
    val_preds = model.predict(X_v)
    val_proba = model.predict_proba(X_v) if hasattr(model, 'predict_proba') else None

    cv_X = X_train_sc if name in ['Logistic Regression', 'KNN'] else X_train
    cv_scores = cross_val_score(model, cv_X, y_train, cv=cv, scoring='accuracy')

    acc = accuracy_score(y_val, val_preds)
    prec = precision_score(y_val, val_preds, average='weighted', zero_division=0)
    rec = recall_score(y_val, val_preds, average='weighted', zero_division=0)
    f1 = f1_score(y_val, val_preds, average='weighted', zero_division=0)
    if val_proba is not None and len(np.unique(y_val)) > 1:
        try:
            auc = roc_auc_score(y_val, val_proba, multi_class='ovr', average='weighted')
        except:
            auc = np.nan
    else:
        auc = np.nan

    results[name] = {
        'model': model, 'val_accuracy': acc, 'val_precision': prec,
        'val_recall': rec, 'val_f1': f1, 'val_auc': auc,
        'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()
    }
    print(f"\n{name}:")
    print(f"  Val Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")
    print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

print("\n" + "=" * 60)
print("STEP 4: MODEL COMPARISON TABLE")
print("=" * 60)

comparison_df = pd.DataFrame([{
    'Model': name,
    'Val Accuracy': f"{r['val_accuracy']:.3f}",
    'Val F1': f"{r['val_f1']:.3f}",
    'Val AUC': f"{r['val_auc']:.3f}" if not np.isnan(r['val_auc']) else 'N/A',
    'CV Mean': f"{r['cv_mean']:.3f}",
    'CV Std': f"±{r['cv_std']:.3f}"
} for name, r in results.items()])
print(comparison_df.to_string(index=False))

best_model_name = max(results, key=lambda n: results[n]['val_f1'])
print(f"\nBest model: {best_model_name} (Val F1 = {results[best_model_name]['val_f1']:.3f})")

print("\n" + "=" * 60)
print("STEP 5: BEST MODEL - TEST EVALUATION")
print("=" * 60)

best = results[best_model_name]
best_model = best['model']
label_names = ['Low', 'Medium', 'High']

if best_model_name in ['Logistic Regression', 'KNN']:
    X_test_eval = X_test_sc
else:
    X_test_eval = X_test

test_preds = best_model.predict(X_test_eval)
test_proba = best_model.predict_proba(X_test_eval)

print(f"\n{best_model_name} - Test Set Results:")
print(f"Accuracy: {accuracy_score(y_test, test_preds):.3f}")
print(f"Precision (weighted): {precision_score(y_test, test_preds, average='weighted'):.3f}")
print(f"Recall (weighted): {recall_score(y_test, test_preds, average='weighted'):.3f}")
print(f"F1 (weighted): {f1_score(y_test, test_preds, average='weighted'):.3f}")
print(f"\nClassification Report:")
print(classification_report(y_test, test_preds, target_names=label_names, zero_division=0))

# PLOTS
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle(f'{best_model_name} — Test Set Evaluation', fontsize=14, fontweight='bold')

# Confusion matrix
cm = confusion_matrix(y_test, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=label_names, yticklabels=label_names,
            linewidths=0.5, cbar=False)
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# ROC Curves (one-vs-rest)
for i, (class_label, color) in enumerate(zip(label_names, COLORS)):
    y_bin = (y_test == i).astype(int)
    fpr, tpr, _ = roc_curve(y_bin, test_proba[:, i])
    auc = roc_auc_score(y_bin, test_proba[:, i])
    axes[1].plot(fpr, tpr, color=color, linewidth=2, label=f'{class_label} (AUC={auc:.2f})')
axes[1].plot([0,1],[0,1],'k--', linewidth=1, alpha=0.5)
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curves (One vs Rest)')
axes[1].legend(fontsize=9)
axes[1].set_facecolor('#F8FAFC')

# Model comparison bar chart
model_names = list(results.keys())
f1_scores = [results[n]['val_f1'] for n in model_names]
bar_colors = [COLORS[0] if n == best_model_name else '#CBD5E1' for n in model_names]
bars = axes[2].barh(model_names, f1_scores, color=bar_colors, edgecolor='white')
axes[2].axvline(baseline_acc, color='#E63946', linewidth=1.5, linestyle='--', label=f'Baseline: {baseline_acc:.2f}')
axes[2].set_xlabel('Validation F1 Score (Weighted)')
axes[2].set_title('Model Comparison')
axes[2].legend(fontsize=9)
for bar, score in zip(bars, f1_scores):
    axes[2].text(score + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{score:.3f}', va='center', fontsize=9)
axes[2].set_facecolor('#F8FAFC')

plt.tight_layout()
plt.savefig('/home/claude/project/4_Modeling/results/model_evaluation.png', bbox_inches='tight', dpi=150)
plt.close()
print("Model evaluation plots saved.")

print("\n" + "=" * 60)
print("STEP 6: FEATURE IMPORTANCE")
print("=" * 60)

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
else:
    perm = permutation_importance(best_model, X_test_eval, y_test, n_repeats=10, random_state=42)
    importances = perm.importances_mean

feat_imp = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)
print(feat_imp.head(12).round(4).to_string())

# Feature importance plot
fig, ax = plt.subplots(figsize=(10, 7))
top_n = feat_imp.head(12)
bar_colors = [COLORS[0] if i < 3 else COLORS[1] if i < 6 else '#CBD5E1' for i in range(len(top_n))]
top_n.sort_values().plot(kind='barh', ax=ax, color=bar_colors[::-1], edgecolor='white')
ax.set_title(f'Top Feature Importances — {best_model_name}', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.set_facecolor('#F8FAFC')
plt.tight_layout()
plt.savefig('/home/claude/project/4_Modeling/results/feature_importance.png', bbox_inches='tight', dpi=150)
plt.close()
print("Feature importance plot saved.")

print("\n" + "=" * 60)
print("STEP 7: SAVE BEST MODEL")
print("=" * 60)

joblib.dump(best_model, '/home/claude/project/4_Modeling/best_model.pkl')
joblib.dump(scaler, '/home/claude/project/4_Modeling/scaler.pkl')
print(f"Best model saved: {best_model_name}")

# Save comparison table
comparison_df.to_csv('/home/claude/project/4_Modeling/results/model_comparison.csv', index=False)
print("Model comparison saved.")
print("\nModeling pipeline complete!")
