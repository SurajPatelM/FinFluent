from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv("transactions_03_2025.csv")
# Use only debit transactions for spending analysis
df_debit = df[df['Transaction Type'] == 'debit'].copy()

# Standardize the amount
scaler = StandardScaler()
df_debit['Amount_scaled'] = scaler.fit_transform(df_debit[['Amount']])

# Fit Isolation Forest
model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
df_debit['outlier_flag'] = model.fit_predict(df_debit[['Amount_scaled']])
df_debit['is_outlier'] = df_debit['outlier_flag'] == -1

# Display outliers
outliers = df_debit[df_debit['is_outlier']]
outliers = outliers.sort_values(by='Amount', ascending=False)

outliers[['Date', 'Description', 'Amount', 'Category']]

output_dir = f"C:/Users/gurpu/Documents/Code/python/generated_data"
file_path = os.path.join(output_dir, f"outliers_2.csv")
outliers.to_csv(file_path, index=False)
