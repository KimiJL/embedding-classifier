import logging
from abc import ABC, abstractmethod
from typing import List, Union

logger = logging.getLogger(__name__)

class Embeddings(ABC):

    @abstractmethod
    def embed_documents(self, data: List[str]) -> List[List[float]]:
        '''embed the documents'''

    @abstractmethod
    def embed_query(self, query: Union[List[str], str]) -> List[List[float]]:
        '''embed the query'''

class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name_or_path: str, device: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("sentence-transformers required for SentenceTransformerEmbeddings")
            raise

        self.model_name_or_path = model_name_or_path
        self.device = device
        self.model = SentenceTransformer(model_name_or_path, device=device)
    
    def meta_data(self):
        return {
            "embedding_type": "SentenceTransformerEmbeddings",
            "model_name_or_path": self.model_name_or_path,
            "device": self.device
        }

    def embed_documents(self, data: List[str]) -> List[List[float]]:
        return self.model.encode(data).tolist()

    def embed_query(self, query: Union[List[str], str]) -> List[List[float]]:
        if isinstance(query, str): 
            query = [query]

        return self.model.encode(query).tolist()
