from abc import ABC, abstractmethod
from typing import List
from sentence_transformers import SentenceTransformer


class Embeddings(ABC):

    @abstractmethod
    def embed_documents(self, data: List[str]) -> List[List[float]]:
        '''embed the documents'''

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        '''embed the query'''

class SentenceTransformerEmbeddings(Embeddings):

    def __init__(self, model_name_or_path: str) -> None:
        self.model = SentenceTransformer(model_name_or_path)
    
    def embed_documents(self, data: List[str]) -> List[List[float]]:
        return self.model.encode(data).tolist()

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query]).tolist()[0]
