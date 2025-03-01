import requests
import pandas as pd

# Read CSV file with predicted stock data
csv_file = "/Users/vidyakalyandurg/Desktop/FinFluent/output1.csv"  # Replace with your actual file path
df = pd.read_csv(csv_file, header=None)  # Assuming no header in the CSV

# Convert predictions to a formatted string
future_stock_prices = ", ".join(df[0].astype(str))

# Define system prompt with integrated stock data
system_prompt = (
    "You are an AI Financial Advisor assistant providing accurate and concise responses. "
    "Here are the predicted stock prices for Apple for the next 30 days: "
    f"{future_stock_prices}. "
    "Use this information to assist users with financial inquiries."
)

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


# Initialize conversation history with updated system prompt
conversation_history = [{"role": "system", "content": system_prompt}]

print("Welcome to your personal AI financial advisor.Get real time stock predictions and uncover valuable insights!\nWhat can I help you with today?\n[We securely handle your financial data to ensure privacy and confidentiality]\n[Type 'exit' to end the conversation]")

while True:
    user_prompt = input("You: ")
    if user_prompt.lower() == "exit":
        print("Goodbye!")
        break

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_prompt})

    # Get AI response
    response = llama3(conversation_history)

    # Add AI response to history
    conversation_history.append({"role": "assistant", "content": response})

    print(f"AI: {response}")
