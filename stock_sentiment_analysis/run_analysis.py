import requests

# Ask user for stock ticker
ticker = input("Enter a stock ticker (e.g., TSLA, AAPL, NVDA): ").strip().upper()

url = f"http://localhost:8000/ticker?ticker={ticker}"

# Send GET request
response = requests.get(url)

# Handle response
if response.ok:
    data = response.json()
    print(f"\n📈 Stock Price: {data.get('price', 'N/A')}")
    print(f"\n🧠 Analysis:\n{data.get('analysis', 'No analysis returned')}\n")

    # print("📰 Sources:")
    # sources = data.get("sources", [])
    # if not sources:
    #     print("No sources returned.")
    # else:
    #     for i, article in enumerate(sources, 1):
    #         url = article.get("url", "No URL")
    #         content_snippet = article.get("content", "")
    #         snippet = content_snippet[:150].strip().replace("\n", " ") + "..." if content_snippet else ""
    #         print(f"{i}. {url}")
    #         if snippet:
    #             print(f"    ↪ {snippet}")
else:
    print("\n❌ Error:", response.status_code)
    print(response.text)
