from typing import Dict
from embedding_classifier.embeddings import Embeddings
from embedding_classifier.model import Model

class EmbeddingClassifier():

    def __init__(self, embedding: Embeddings, model: Model, model_name: str = "default") -> None:
        # embedding model to use
        self.embedding = embedding
        self.models : Dict[str, Model] = {
            model_name: model
        }

    # should train the model
    def fit(self, train, pred, model_name: str = "default"):
        train_embeddings = []

        train_embeddings = self.embedding.embed_documents(train)
        
        self.models[model_name].fit(train_embeddings, pred)

    # make prediction
    def predict(self, data, model_name: str = "default"):
        # TODO check that it's a string vs list, shouldn't iterate through string
        query = self.embedding.embed_query(data)

        return self.models[model_name].predict(query)

    def save_model(self, save_path: str, model_name: str = "default"):
        self.models[model_name].save(save_path)
