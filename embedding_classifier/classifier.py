from embedding_classifier.embeddings import Embeddings
from embedding_classifier.model import Model

class EmbeddingClassifier():

    def __init__(self, embedding: Embeddings, model: Model) -> None:
        # embedding model to use
        self.embedding = embedding
        self.model = model

    # should train the model
    def fit(self, train, pred):
        train_embeddings = []

        train_embeddings = self.embedding.embed_documents(train)
        
        self.model.fit(train_embeddings, pred)

    # make prediction
    def predict(self, data):
        # TODO check that it's a string vs list, shouldn't iterate through string
        query = (self.embedding.embed_query(data))

        return self.model.predict(query)

    def save_model(self, save_path: str):
        self.model.save(save_path)
