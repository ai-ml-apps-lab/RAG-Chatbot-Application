
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import logging
import config


class LLMManager:
    """
    Handles model selection, temperature settings, and provides
    LLM and embedding instances for the RAG pipeline.
    """
    def __init__(self):
        self._llm = None
        self._embedding = None
        self.logger = logging.getLogger(__name__)

    def set_llm(self, model_name: None, temperature=None):
        self.logger.info(f"Switching LLM model to: {model_name} and temperature to: {temperature}")
        self._llm = OpenAI(
            model=model_name or config.LLM_MODEL_ID,
            api_key=config.OPENAI_API_KEY,
            max_tokens=config.MAX_TOKENS,
            temperature=temperature if temperature is not None else config.TEMPERATURE
        )
        Settings.llm = self._llm
        return self._llm


    def get_embedding(self):
        if self._embedding is None:
            self._embedding = OpenAIEmbedding(
                model=config.EMBEDDING_MODEL_ID,
                api_key=config.OPENAI_API_KEY,
            )
        return self._embedding
    
