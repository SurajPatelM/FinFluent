import pandas as pd
import glob
import re
import os  # Needed to delete files

# Step 1: Get all matching CSV files
files = glob.glob("transactions_*.csv")


# Step 2: Extract month and year from filename, then sort
def extract_date(filename):
    match = re.search(r'transactions_(\d{2})_(\d{4})', filename)
    if match:
        month, year = match.groups()
        return int(year), int(month)
    else:
        return float('inf'), float('inf')  # Put unmatched files at the end


# Sort files by extracted date
sorted_files = sorted(files, key=extract_date)

# Step 3: Load and concatenate all files
df_list = [pd.read_csv(f) for f in sorted_files]
merged_df = pd.concat(df_list, ignore_index=True)

user_number = 5
user_number = str(user_number)
# Step 4: Save merged result
output_file = "user_",user_number,+".csv"
merged_df.to_csv(output_file, index=False)

# Step 5: Delete original files (except the merged one)
for f in sorted_files:
    if f != output_file:
        os.remove(f)
