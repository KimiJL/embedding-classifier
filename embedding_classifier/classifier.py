from embedding_classifier.embeddings import Embeddings
from embedding_classifier.model import Model

class EmbeddingClassifier():

    def __init__(self, embedding: Embeddings, model: Model) -> None:
        # embedding model to use
        self.embedding = embedding
        self.model = model

    # should train the model
    def fit(self, train, pred):
        # TODO batch encode
        train = []
        for t in train:
            train.append(self.embedding.embed_documents([t])[0])
        
        self.model.fit(train, pred)

    # make prediction
    def predict(self, data):
        query = []
        for d in data:
            query.append(self.embedding.embed_query(d))

        self.model.predict(query)