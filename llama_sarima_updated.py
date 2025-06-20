import requests
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import MonthEnd

# Load the CSV file
df = pd.read_csv(
    "C:\\Users\\aksha\\Downloads\\FinFluent-main\\FinFluent-main\\sorted_transactions.csv",
    parse_dates=["Date"],
)

debit_categories = {
    "Shopping",
    "Entertainment",
    "Bills",
    "Restaurants",
    "Travel",
    "Mortgage & Rent",
}

df = df[df["Category"].isin(debit_categories)]
df["Amount"] = df["Amount"].abs()
df["Month"] = (df["Date"] + MonthEnd(0)).dt.to_period("M").dt.to_timestamp()
monthly_spending = df.groupby(["Month", "Category"])["Amount"].sum().unstack().fillna(0)

# Ensure the index has explicit frequency
monthly_spending.index = pd.date_range(
    start=monthly_spending.index.min(), periods=len(monthly_spending), freq="MS"
)


# Function to train SARIMA and forecast next month
def forecast_sarima(data, steps=1):
    model = SARIMAX(
        data,
        order=(3, 0, 0),
        seasonal_order=(1, 0, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    model_fit = model.fit(disp=False)
    return model_fit.forecast(steps=steps)


# Generate forecast
future_spending = {}
for category in monthly_spending.columns:
    forecast = forecast_sarima(monthly_spending[category])
    future_spending[category] = forecast.iloc[0]

# Generate system prompt with forecasted spending
forecast_text = "You are given information on a user's predicted spending for next month:\n"
for category, amount in future_spending.items():
    forecast_text += f"{category}: ${amount:.2f}\n"
# print(forecast_text)

system_prompt = f"""
You are an AI-powered Financial Advisor. Your job is to provide **accurate, data-driven financial guidance**.

## User Information:
- **Monthly Salary:** $10,000  
- **Predicted Spending for Next Month:**
{forecast_text}

## Instructions:
1. **Be Specific & Data-Driven:**  
   - When answering questions, refer to the user’s salary and predicted spending.  
   - Use **numbers, percentages, and trends** instead of generic advice.  

2. **Provide Cost-Saving Strategies:**  
   - If spending in a category is **higher than 30% of salary**, suggest ways to reduce it.  
   - Recommend savings and investment strategies based on spending habits.  

3. **Warn About Potential Issues:**  
   - If spending in a category has **increased significantly**, explain why it might be a problem.  
   - Identify risky spending patterns.


"""

# LLaMA 3 Chatbot setup
url = "http://localhost:11434/api/chat"


def llama3(conversation_history):
    data = {
        "model": "llama3",
        "messages": conversation_history,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=data)
    return response.json()["message"]["content"]


conversation_history = [{"role": "system", "content": system_prompt}]

print("Welcome to your personal AI financial advisor.\nDiscover your projected monthly spending and gain deeper insights!\nWhat can I help you with today?\n[We securely handle your financial data to ensure privacy and confidentiality]\n[Type 'exit' to end the conversation]")

while True:
    user_prompt = input("You: ")
    if user_prompt.lower() == "exit":
        print("Goodbye!")
        break

    conversation_history.append({"role": "user", "content": user_prompt})
    response = llama3(conversation_history)
    conversation_history.append({"role": "assistant", "content": response})
    print(f"FinFluent: {response}")