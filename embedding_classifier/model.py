from abc import ABC, abstractmethod
from typing import List

from sklearn.ensemble import RandomForestClassifier

class Model(ABC):

    @abstractmethod
    def fit(self, train: List[List[float]], pred: List[int]):
        pass

    @abstractmethod
    def predict(self, input: List[List[float]]):
        pass


class RandomForest(Model):

    def __init__(self) -> None:
        self.model = RandomForestClassifier()
    
    def fit(self, train: List[List[float]], pred: List[int]):
        self.model.fit(train, pred)

    def predict(self, input: List[List[float]]):
        return self.model.predict(input)