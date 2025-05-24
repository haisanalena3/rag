import os
from dotenv import load_dotenv

def load_config():
    load_dotenv()
    config = {
        "URLS": os.getenv("URLS", "").split(","),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "gemini-pro")
    }
    return config