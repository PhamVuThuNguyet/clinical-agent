import os

from dotenv import load_dotenv

load_dotenv()


class LLMConstants:
    GPT_MODEL = "gpt-4o"
    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
