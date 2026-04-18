#!/usr/bin/env python3
"""
# DATA CLEANING & PREPARATION NOTEBOOK
# CSC 3221 - Introduction to Data Science Course
# ICT University Cameroon
# Mobile Money Transaction Analysis

## Section 1: Setup and Data Loading
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'primary': '#028090', 'secondary': '#02C39A', 'accent': '#F0A500',
          'danger': '#E63946', 'neutral': '#6B7280'}

os.makedirs('/home/claude/project/2_Data_Cleaning', exist_ok=True)
os.makedirs('/home/claude/project/3_EDA/visualizations', exist_ok=True)

print("=" * 60)
print("STEP 1: LOADING RAW DATA")
print("=" * 60)

tx_raw = pd.read_csv('/home/claude/project/1_Data_Collection/raw_transactions.csv')
demo_raw = pd.read_csv('/home/claude/project/1_Data_Collection/raw_demographics.csv')

print(f"Raw transactions shape: {tx_raw.shape}")
print(f"Raw demographics shape: {demo_raw.shape}")
print(f"\nTransaction columns: {list(tx_raw.columns)}")
print(f"\nTransaction dtypes:\n{tx_raw.dtypes}")

##############################################################
print("\n" + "=" * 60)
print("STEP 2: DATA QUALITY ASSESSMENT")
print("=" * 60)

print("\n--- Missing Values (Transactions) ---")
missing_tx = tx_raw.isnull().sum()
pct_missing_tx = (tx_raw.isnull().sum() / len(tx_raw) * 100).round(2)
quality_report_tx = pd.DataFrame({
    'Missing Count': missing_tx,
    'Missing %': pct_missing_tx
})
print(quality_report_tx[quality_report_tx['Missing Count'] > 0])

print("\n--- Missing Values (Demographics) ---")
missing_demo = demo_raw.isnull().sum()
print(missing_demo[missing_demo > 0] if missing_demo.sum() > 0 else "No missing values in demographics")

print("\n--- Duplicate Rows ---")
dup_tx = tx_raw.duplicated().sum()
print(f"Duplicate transactions: {dup_tx}")

print("\n--- Outlier/Error Detection ---")
amt_notna = tx_raw['amount_xaf'].dropna()
neg_count = (amt_notna < 0).sum()
zero_count = (amt_notna == 0).sum()
print(f"Negative amounts: {neg_count}")
print(f"Zero amounts: {zero_count}")

Q1 = amt_notna.quantile(0.25)
Q3 = amt_notna.quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
outliers = ((amt_notna < lower_fence) | (amt_notna > upper_fence)).sum()
print(f"IQR outliers (amount): {outliers} ({outliers/len(amt_notna)*100:.1f}%)")
print(f"IQR Fence: [{lower_fence:.0f}, {upper_fence:.0f}] XAF")

print("\n--- Value Consistency Check ---")
print(f"Transaction status values: {tx_raw['status'].value_counts().to_dict()}")
print(f"Transaction type nulls: {tx_raw['transaction_type'].isnull().sum()}")

##############################################################
print("\n" + "=" * 60)
print("STEP 3: DATA CLEANING")
print("=" * 60)

tx = tx_raw.copy()

# 3a. Remove duplicates
before = len(tx)
tx = tx.drop_duplicates()
print(f"Duplicates removed: {before - len(tx)} rows")

# 3b. Fix negative amounts (absolute value - likely sign error)
neg_mask = tx['amount_xaf'] < 0
tx.loc[neg_mask, 'amount_xaf'] = tx.loc[neg_mask, 'amount_xaf'].abs()
print(f"Negative amounts corrected: {neg_mask.sum()}")

# 3c. Handle missing amounts - impute with median by transaction_type
print("\nImputing missing amounts by transaction type median...")
before_null = tx['amount_xaf'].isnull().sum()
tx['amount_xaf'] = tx.groupby('transaction_type')['amount_xaf'].transform(
    lambda x: x.fillna(x.median())
)
# If tx_type also null, use global median
tx['amount_xaf'] = tx['amount_xaf'].fillna(tx['amount_xaf'].median())
print(f"Amounts imputed: {before_null}")

# 3d. Handle missing transaction types - impute with mode
before_null_type = tx['transaction_type'].isnull().sum()
tx['transaction_type'] = tx['transaction_type'].fillna(tx['transaction_type'].mode()[0])
print(f"Transaction types imputed: {before_null_type}")

# 3e. Parse and standardize dates
tx['transaction_date'] = pd.to_datetime(tx['transaction_date'])
tx['transaction_month'] = pd.to_datetime(tx['transaction_month'])
tx['month_num'] = tx['transaction_date'].dt.month
tx['year'] = tx['transaction_date'].dt.year
tx['quarter'] = tx['transaction_date'].dt.quarter

# 3f. IQR outlier treatment - cap at fences (Winsorizing)
Q1 = tx['amount_xaf'].quantile(0.25)
Q3 = tx['amount_xaf'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers_before = ((tx['amount_xaf'] < lower) | (tx['amount_xaf'] > upper)).sum()
tx['amount_xaf'] = tx['amount_xaf'].clip(lower=lower, upper=upper)
print(f"Outliers winsorized (IQR method): {outliers_before}")

# Filter to successful transactions for feature engineering
tx_success = tx[tx['status'] == 'Success'].copy()
print(f"\nSuccessful transactions: {len(tx_success)} / {len(tx)}")

##############################################################
print("\n" + "=" * 60)
print("STEP 4: FEATURE ENGINEERING")
print("=" * 60)

# Per-user aggregate features
user_features = tx_success.groupby('user_id').agg(
    total_transactions=('transaction_id', 'count'),
    total_amount_sent=('amount_xaf', lambda x: x[tx_success.loc[x.index, 'transaction_type'].isin(
        ['Send Money', 'Pay Bill', 'Buy Airtime', 'Merchant Payment', 'Withdraw Cash'])].sum()),
    total_amount_received=('amount_xaf', lambda x: x[tx_success.loc[x.index, 'transaction_type'].isin(
        ['Receive Money', 'Deposit Cash'])].sum()),
    avg_transaction_amount=('amount_xaf', 'mean'),
    max_transaction_amount=('amount_xaf', 'max'),
    std_transaction_amount=('amount_xaf', 'std'),
    total_fees_paid=('fee_xaf', 'sum'),
    weekend_transactions=('is_weekend', 'sum'),
    first_tx_date=('transaction_date', 'min'),
    last_tx_date=('transaction_date', 'max'),
).reset_index()

# Send/Receive ratio
eps = 1e-6
user_features['send_receive_ratio'] = (
    user_features['total_amount_sent'] / (user_features['total_amount_received'] + eps)
).round(3)

# Weekend ratio
user_features['weekend_ratio'] = (
    user_features['weekend_transactions'] / user_features['total_transactions']
).round(3)

# Active months
user_features['first_tx_date'] = pd.to_datetime(user_features['first_tx_date'])
user_features['last_tx_date'] = pd.to_datetime(user_features['last_tx_date'])
user_features['active_months'] = (
    (user_features['last_tx_date'] - user_features['first_tx_date']).dt.days / 30
).clip(lower=1).round(1)

# Transactions per month
user_features['tx_per_month'] = (
    user_features['total_transactions'] / user_features['active_months']
).round(2)

# Days since last transaction
ref_date = pd.Timestamp('2025-03-01')
user_features['days_since_last_tx'] = (
    ref_date - user_features['last_tx_date']
).dt.days

# Monthly average amount
user_features['avg_monthly_amount'] = (
    user_features['avg_transaction_amount'] * user_features['tx_per_month']
).round(0)

# Transaction diversity (unique tx types)
tx_diversity = tx_success.groupby('user_id')['transaction_type'].nunique().reset_index()
tx_diversity.columns = ['user_id', 'tx_type_diversity']
user_features = user_features.merge(tx_diversity, on='user_id', how='left')

# Most common transaction type
most_common_type = tx_success.groupby('user_id')['transaction_type'].agg(
    lambda x: x.mode()[0]).reset_index()
most_common_type.columns = ['user_id', 'most_common_tx_type']
user_features = user_features.merge(most_common_type, on='user_id', how='left')

print(f"User features engineered: {list(user_features.columns)}")
print(f"\nFeature statistics:\n{user_features[['total_transactions','tx_per_month','avg_transaction_amount']].describe().round(0)}")

##############################################################
print("\n" + "=" * 60)
print("STEP 5: DATA TRANSFORMATION & FINAL DATASET")
print("=" * 60)

# Merge with demographics
demo = demo_raw.copy()
final_df = demo.merge(user_features, on='user_id', how='left')

# Encode categorical variables
# Label encoding for ordinal
edu_order = {'No formal education': 0, 'Primary': 1, 'Secondary': 2, 'Vocational': 3, 'University': 4}
income_order = {'Below 50,000 XAF': 0, '50,000-100,000 XAF': 1, '100,000-200,000 XAF': 2,
                '200,000-500,000 XAF': 3, 'Above 500,000 XAF': 4}
zone_order = {'rural': 0, 'suburban': 1, 'urban': 2}
segment_order = {'Low': 0, 'Medium': 1, 'High': 2}

final_df['education_encoded'] = final_df['education_level'].map(edu_order)
final_df['income_encoded'] = final_df['monthly_income_range'].map(income_order)
final_df['zone_encoded'] = final_df['zone_type'].map(zone_order)
final_df['activity_label'] = final_df['activity_segment'].map(segment_order)
final_df['gender_encoded'] = (final_df['gender'] == 'Male').astype(int)
final_df['smartphone_encoded'] = (final_df['smartphone_owner'] == 'Yes').astype(int)

# One-hot for nominal
final_df = pd.get_dummies(final_df, columns=['primary_mm_use', 'mm_provider'], drop_first=False)

# Drop helper columns
drop_cols = ['first_tx_date', 'last_tx_date', 'total_amount_sent', 'total_amount_received',
             'activity_score_raw', 'most_common_tx_type']
final_df = final_df.drop(columns=[c for c in drop_cols if c in final_df.columns])

print(f"Final dataset shape: {final_df.shape}")
print(f"Target distribution:\n{final_df['activity_segment'].value_counts()}")

# Save cleaned datasets
tx.to_csv('/home/claude/project/2_Data_Cleaning/cleaned_transactions.csv', index=False)
final_df.to_csv('/home/claude/project/2_Data_Cleaning/cleaned_data.csv', index=False)
user_features.to_csv('/home/claude/project/2_Data_Cleaning/user_features.csv', index=False)

##############################################################
print("\n" + "=" * 60)
print("STEP 6: BEFORE/AFTER SUMMARY STATISTICS")
print("=" * 60)

print("\nBEFORE cleaning - amount_xaf stats:")
print(tx_raw['amount_xaf'].describe().round(0))
print("\nAFTER cleaning - amount_xaf stats:")
print(tx['amount_xaf'].describe().round(0))

print("\nData cleaning complete. Files saved.")
