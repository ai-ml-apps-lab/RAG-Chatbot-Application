from llama_index.core.memory import ChatMemoryBuffer
import uuid
import config


class RAGSession:
    """
    Handles summarization and question-answering using
    the associated vector index.
    """
    def __init__(self, index):
        self.index = index
        self.memory = ChatMemoryBuffer.from_defaults(token_limit=2000)

    def summarize(self):

        query_engine = self.index.as_query_engine(
            similarity_top_k = config.SIMILARITY_TOP_K
        )

        prompt = "Provide a clear and concise summary."

        response = query_engine.query(prompt)
        return str(response)

    def chat(self, query):

        query_engine = self.index.as_query_engine(
            similarity_top_k = config.SIMILARITY_TOP_K,
            memory=self.memory
        )

        response = query_engine.query(query)
        return str(response)
    

class SessionManager:

    def __init__(self):
        self.sessions = {}

    def create_session(self, session):
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = session
        return session_id, session
    
    def get_session(self, session_id):
        return self.sessions.get(session_id)
    
    