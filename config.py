"""
Central configuration for the LlamaIndex RAG Chatbot (OpenAI version)
"""

from dotenv import load_dotenv
import os

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Models
LLM_MODEL_ID = "gpt-4o-mini"

AVAILABLE_LLM_MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
]

EMBEDDING_MODEL_ID = "text-embedding-3-small"


# Generation Parameters
TEMPERATURE = 0.3
MAX_TOKENS = 512

# RAG Parameters
CHUNK_SIZE = 500
SIMILARITY_TOP_K = 5