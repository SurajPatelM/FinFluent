import requests

url = "http://localhost:11434/api/chat"


def llama3(system_prompt, conversation_history):
    data = {
        "model": "llama3",
        "messages": conversation_history,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, headers=headers, json=data)
    return response.json()["message"]["content"]


# Initialize conversation history
conversation_history = [
    {
        "role": "system",
        "content": """You are an AI Financial Advisor assistant providing accurate and concise responses.\n
        You are given some outliers in the user's transactions:
        The transaction of the the amount $4199.87 dated on 2023-01-15 under Walmart is flagged as an anomaly because it is an unusually high-value transaction based on the category trend.
        The transaction of the the amount $1238.2 dated on 2023-03-11 under Target is flagged as an anomaly because it is an unusually high-value transaction indicating a sudden large purchase.
        The transaction of the the amount $203.8 dated on 2023-10-13 under AMC Theatres is flagged as an anomaly because it is an unusually high-value transaction in a non-recurring category due to extreme spending on luxury items.""",
    }
]

print("Welcome to your personal AI financial advisor.\nIdentify any unusual spending patterns and uncover valuable insights!\nWhat can I help you with today?\n[We securely handle your financial data to ensure privacy and confidentiality]\n[Type 'exit' to end the conversation]")

while True:
    user_prompt = input("You: ")
    if user_prompt.lower() == "exit":
        print("Goodbye!")
        break

    # Add user message to history
    conversation_history.append({"role": "user", "content": user_prompt})

    # Get AI response
    response = llama3("AI Financial Advisor Assistant", conversation_history)

    # Add AI response to history
    conversation_history.append({"role": "assistant", "content": response})

    print(f"AI: {response}")
