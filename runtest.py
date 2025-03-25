import os
from dotenv import load_dotenv
load_dotenv()
api_key = os.environ.get("INDIANKANOON_API_KEY")
print(f"API Key: {api_key}")
