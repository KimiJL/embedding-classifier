from abc import ABC, abstractmethod
from typing import Dict, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans as SklearnKMeans

class Model(ABC):

    @abstractmethod
    def fit(self, train: List[List[float]], pred: List[int] = None):
        pass

    @abstractmethod
    def predict(self, input: List[List[float]]):
        pass


class RandomForest(Model):

    def __init__(self, model_args: Dict = None) -> None:
        model_args = model_args if model_args else {}
        self.model = RandomForestClassifier(**model_args)
    
    def fit(self, train: List[List[float]], pred: List[int] = None):
        if pred is None:
            raise Exception("pred is required for supervised learning")

        self.model.fit(train, pred)

    def predict(self, input: List[List[float]]):
        return self.model.predict(input)
    
class KMeans(Model):
    def __init__(self, model_args: Dict = None) -> None:
        model_args = model_args if model_args else {}
        self.model = SklearnKMeans(**model_args)

    def fit(self, train: List[List[float]], pred: List[int]= None):
        self.model.fit(train)
    
    def predict(self, input: List[List[float]]):
        return self.model.predict(input)