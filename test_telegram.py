from dotenv import load_dotenv
import os
import requests

load_dotenv()

token = os.environ.get("TELEGRAM_BOT_TOKEN")
chat_id = os.environ.get("TELEGRAM_CHAT_ID")

print(f"Token found: {'YES' if token else 'NO'}")
print(f"Chat ID found: {'YES' if chat_id else 'NO'}")

if not token or not chat_id:
    print("\nERROR: Check your .env file - one or both values are missing")
else:
    r = requests.post(
        f"https://api.telegram.org/bot{token}/sendMessage",
        json={"chat_id": chat_id, "text": "IPL bot test - working!"}
    )
    print(f"\nTelegram response: {r.json()}")
