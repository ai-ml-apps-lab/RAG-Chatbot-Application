from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.settings import Settings
import logging
import config


class VectorStoreManager:
    """
    Converts raw text into embeddings and builds a vector store index 
    for retrieval in the RAG pipeline.
    """
    def __init__(self, embedding_model):
        Settings.embed_model = embedding_model
        self.logger = logging.getLogger(__name__)

    def build_index_from_text(self, text: str):
        try:
            document = Document(text=text)
            splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE)
            nodes = splitter.get_nodes_from_documents([document])
            index = VectorStoreIndex(nodes)
            self.logger.info(f"Vector Store was created.")
            return index        
        except Exception as e:
            self.logger.error(f"Vector Store error: {e}")
            return []
