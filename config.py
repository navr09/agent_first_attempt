from dotenv import load_dotenv
import os

load_dotenv()  # Load .env file

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
