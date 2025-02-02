import requests
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.tseries.offsets import MonthEnd
from cryptography.fernet import Fernet

# Use a fixed encryption key (store securely in production)
key = Fernet.generate_key()
cipher_suite = Fernet(key)

# Function to encrypt data
def encrypt_data(data):
    """Encrypts the given data string only if it's not already encrypted."""
    if isinstance(data, str) and data.startswith("gAAAAA"):  # Fernet encrypted strings start with this prefix
        return data
    encrypted_data = cipher_suite.encrypt(data.encode())
    return encrypted_data.decode()

# Function to decrypt data
def decrypt_data(encrypted_data):
    """Decrypts the encrypted data string if it's encrypted."""
    if not isinstance(encrypted_data, str) or not encrypted_data.startswith("gAAAAA"):
        return encrypted_data  # Return plaintext as-is
    decrypted_data = cipher_suite.decrypt(encrypted_data.encode())
    return decrypted_data.decode()

# Load the CSV file
df = pd.read_csv(
    "/Users/vidyakalyandurg/Desktop/FinFluent/synthetic_bank_statements_user_5y1.csv",
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
forecast_text = "Predicted spending for next month:\n"
for category, amount in future_spending.items():
    forecast_text += f"{category}: ${amount:.2f}\n"

# Encrypt forecast before adding to system prompt
encrypted_forecast_text = encrypt_data(forecast_text)

system_prompt = (
    """You are an AI Financial Advisor assistant providing accurate and concise responses. 
    We securely handle your financial data to ensure privacy and confidentiality.\n\n"""
    + decrypt_data(encrypted_forecast_text)  # Decrypt before using in prompt
)

# LLaMA 3 Chatbot setup
url = "http://localhost:11434/api/chat"

def llama3(conversation_history):
    """Sends encrypted messages to LLaMA 3 API and gets a response."""
    
    # Encrypt conversation history before sending
    encrypted_history = [ 
        {"role": entry["role"], "content": encrypt_data(entry["content"])} 
        for entry in conversation_history
    ]
    
    data = {
        "model": "llama3",
        "messages": encrypted_history,
        "stream": False,
    }
    
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, headers=headers, json=data)

    # AI responses are NOT encrypted, so return them directly
    return response.json().get("message", {}).get("content", "Error: No response received.")

# Start conversation
conversation_history = [{"role": "system", "content": system_prompt}]

print("Welcome to your personal AI financial advisor. Discover your projected monthly spending and gain deeper insights!")
print("We securely handle your financial data for privacy and confidentiality. Type 'exit' to end the conversation.")

while True:
    user_prompt = input("You: ")
    if user_prompt.lower() == "exit":
        print("Goodbye!")
        break

    # Encrypt user input before storing but decrypt before sending to AI
    encrypted_user_prompt = encrypt_data(user_prompt)
    
    conversation_history.append({"role": "user", "content": decrypt_data(encrypted_user_prompt)})  # Decrypt before using

    response = llama3(conversation_history)
    conversation_history.append({"role": "assistant", "content": response})
    print(f"AI: {response}")
