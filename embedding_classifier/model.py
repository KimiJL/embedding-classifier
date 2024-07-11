from abc import ABC, abstractmethod
from typing import Dict, List
from pickle import dump, HIGHEST_PROTOCOL

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans as SklearnKMeans

class Model(ABC):

    def __init__(self, model_args: Dict = None) -> None:
        self.model_args = model_args
        self.fitted_model = None
        self.embedding_size = None
    
    def fit(self, train: List[List[float]], pred: List[int] = None):
        if not isinstance(train, list) and len(train) > 0:
            raise Exception("Expect list of embeddings for training")
        
        if not isinstance(train[0], list):
            raise Exception("Embedding in train set is not a list")
    
        self.embedding_size = len(train[0])
        self._fit(train, pred)

    @abstractmethod
    def _fit(self, train: List[List[float]], pred: List[int] = None):
        pass

    @abstractmethod
    def predict(self, input: List[List[float]]):
        pass
    
    @abstractmethod
    def save(self, save_path: str):
        pass


class RandomForest(Model):
    
    def _fit(self, train: List[List[float]], pred: List[int] = None):
        if pred is None:
            raise Exception("pred is required for supervised learning")
        
        self.fitted_model = RandomForestClassifier(**self.model_args)
        self.fitted_model.fit(train, pred)

    def predict(self, input: List[List[float]]):
        if self.fitted_model is None:
            raise Exception("model is not trained or loaded for prediction")

        return self.fitted_model.predict(input)

    def save(self, save_path: str):
        with open(save_path, "wb") as f:
            dump(self.fitted_model, f, protocol=HIGHEST_PROTOCOL)
    
class KMeans(Model):

    def _fit(self, train: List[List[float]], pred: List[int]= None):
        self.fitted_model = SklearnKMeans(**self.model_args)

        self.fitted_model.fit(train)
    
    def predict(self, input: List[List[float]]):
        if self.fitted_model is None:
            raise Exception("model is not trained or loaded for prediction")

        return self.fitted_model.predict(input)

    def save(self, save_path: str):
        with open(save_path, "wb") as f:
            dump(self.fitted_model, f, protocol=HIGHEST_PROTOCOL)