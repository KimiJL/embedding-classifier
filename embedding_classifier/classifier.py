from embedding_classifier.embeddings import Embeddings
from sklearn.ensemble import RandomForestClassifier

class EmbeddingClassifier():

    def __init__(self, embedding: Embeddings) -> None:
        # embedding model to use
        self.embedding = embedding
        self.model = None

    # should train the model
    def fit(self, train, pred):
        # TODO batch encode
        train = []
        for t in train:
            train.append(self.embedding.embed_documents([t])[0])
        
        self.model = RandomForestClassifier()
        self.model.fit(train, pred)

    # make prediction
    def predict(self, data):
        query = []
        for d in data:
            query.append(self.embedding.embed_query(d))

        self.model.predict(query)