from datetime import datetime, timedelta
import pandas as pd
import random
import os
from calendar import monthrange

salary = 6000
rent = 1400

# Global variable for salary tracking
current_salary = round(random.uniform(salary, salary), 2)
current_rent = round(random.uniform(rent, rent), 2)
# Function to modify debit category ranges slightly
def adjust_ranges(base_range, variation=10):
    min_val, max_val = base_range
    new_min = max(1, min_val + random.randint(-variation, variation))
    new_max = max(new_min, max_val + random.randint(-variation, variation))
    return (new_min, new_max)

def salaryAdjustment(salary):
    if random.choices([True, False], weights=[0.2, 0.8])[0]:
        hike = random.choices([1.2, 1.5, 1.4, 1.3], weights=[0.25]*4)[0]
    else:
        hike = random.choices([1.02, 1.03, 1.04, 1.1], weights=[0.25, 0.55, 0.15, 0.05])[0]
    print(salary)
    return round(salary * hike, 0)

def inflationPercent(year):
    if year == 2020:
        return 1
    return random.choices([1.02, 1.04, 1.05, 1.06], weights=[0.1, 0.4, 0.3, 0.2])[0]

def adjustRent(rent):
    hike = random.choices([50, 100, 250, 300], weights= [0.35, 0.4, 0.15, 0.1], k=1)[0]
    hike = float(hike)
    return round(rent+hike, 2)

# Category Definitions
debit_categories = [
    {"name": "Entertainment", "prob": 0.15, "data": {"Entertainment": (20, 70)}, "desc": ["AMC Boston", "StandUp show", "Karaoke", "Ice skating"]},
    {"name": "Restaurants", "prob": 0.15, "data": {"Restaurants": (25, 80)}, "desc": ["Boston Halal", "Boston Shawarma", "Dominos", "Mama-cita"]},
    {"name": "Grocery shopping", "prob": 0.45, "data": {"Grocery shopping": (150, 250)}, "desc": ["Trader Joes", "Stop and Shop", "Ace Hardware", "Market Basket"]},
    {"name": "Travel expenses", "prob": 0.15, "data": {"Travel expenses": (10, 30)}, "desc": ["Uber", "Lyft", "MBTA"]},
    {"name": "Shopping", "prob": 0.1, "data": {"Shopping": (20, 80)}, "desc": ["Amazon", "Mall", "Temu"]},
]

credit_categories = [
    {"name": "Dividends", "prob": 0.25, "data": {"Dividends": (50, 200)}},
    {"name": "Cashback Rewards", "prob": 0.75, "data": {"Cashback Rewards": (3, 20)}}
]

# Fixed values
#rent_debit = {"Rent/ Mortgage": (2400, 2400)}
util_debit = {"Utilities": (50, 80)}
heating_util_debit = {"Heating fuel": (150, 250)}
salary_credit = {"Salary": (10000, 10000)}
account_names = ["Credit Card", "Checking", "Savings"]

# Transaction generator
def generate_transactions(month, year):
    global current_salary
    global current_rent
    transactions = []
    used_descriptions_by_date = {}
    start_date = datetime(year, month, 1)
    last_day = monthrange(year, month)[1]

    # Apply raise only in November
    if month == 11:
        current_salary = salaryAdjustment(current_salary)

    if month == 9:
        current_rent = adjustRent(current_rent)


    inflation = inflationPercent(year)
    salary_date = start_date + timedelta(days=12)
    rent_date = start_date + timedelta(days=0)
    transactions.append([salary_date.strftime('%m/%d/%Y'), f"Monthly direct deposit - Salary", round(current_salary, 2), "credit", "Salary", "Savings"])
    transactions.append([rent_date.strftime('%m/%d/%Y'), f"Rent/ Mortgage", round(current_rent, 2), "debit", "Mortgage & Rent", "Savings"])
    rent_date = start_date
    transactions.extend([
        #[rent_date.strftime('%m/%d/%Y'), "Rent", round(random.uniform(*rent_debit["Rent/ Mortgage"]), 2), "debit", "Mortgage & Rent", "Checking"],
        [rent_date.strftime('%m/%d/%Y'), "Eversource Electricity", round(random.uniform(*util_debit["Utilities"]), 2), "debit", "Utilities", "Checking"],
        [rent_date.strftime('%m/%d/%Y'), "National Grid Gas", round(random.uniform(*util_debit["Utilities"]), 2), "debit", "Utilities", "Checking"]
    ])

    if month in [10, 11, 12, 1, 2, 3, 4]:
        transactions.append([rent_date.strftime('%m/%d/%Y'), "Heating fuel", round(random.uniform(*heating_util_debit["Heating fuel"]), 2), "debit", "Utilities", "Checking"])

    num_records = random.randint(70, 120)
    for _ in range(num_records):
        date_obj = start_date + timedelta(days=random.randint(0, last_day - 1))
        date_str = date_obj.strftime('%m/%d/%Y')
        transaction_type = random.choices(["debit", "credit"], weights=[0.95, 0.05])[0]

        if transaction_type == "debit":
            cat = random.choices(debit_categories, weights=[c["prob"] for c in debit_categories])[0]
            category = list(cat["data"].keys())[0]
            available_desc = [d for d in cat["desc"] if date_str not in used_descriptions_by_date or d not in used_descriptions_by_date[date_str]]
            if not available_desc:
                continue
            description = random.choice(available_desc)
            used_descriptions_by_date.setdefault(date_str, set()).add(description)
            amount = round(random.uniform(*cat["data"][category]) * inflation, 2)
            account_name = "Credit Card"
        else:
            cat = random.choices(credit_categories, weights=[c["prob"] for c in credit_categories])[0]
            category = list(cat["data"].keys())[0]
            amount = round(random.uniform(*cat["data"][category]), 2)
            description = "Credit Transaction"
            account_name = "Savings"

        transactions.append([date_str, description, round(amount, 2), transaction_type, category, account_name])

    return pd.DataFrame(transactions, columns=["Date", "Description", "Amount", "Transaction Type", "Category", "Account Name"])

# Output directory
output_dir = "generated_data"
os.makedirs(output_dir, exist_ok=True)

# Generate data for 2020-2024
for year in range(2020, 2025):
    for month in range(1, 13):
        df_transactions = generate_transactions(month, year)
        df_transactions["Date"] = pd.to_datetime(df_transactions["Date"]).dt.strftime('%m-%d-%Y')
        df_transactions = df_transactions.sort_values(by="Date", ascending=True)
        file_path = os.path.join(output_dir, f"transactions_{month:02d}_{year}.csv")
        df_transactions.to_csv(file_path, index=False)
        print(f"Data for {month:02d}/{year} generated successfully!")

month = 1
year = 2025

df_test = generate_transactions(month, year)
df_test["Date"] = pd.to_datetime(df_test["Date"]).dt.strftime('%m-%d-%Y')
df_test = df_test.sort_values(by="Date", ascending=True)
file_path = os.path.join(output_dir, f"test_data_{month:02d}_{year}.csv")
df_test.to_csv(file_path, index=False)
print(f"Data for {month:02d}/{year} generated successfully!")