from abc import abstractmethod, ABC
from collections import Counter
from dataclasses import dataclass
import numpy as np
from pandas import DataFrame, Series


class DistanceMeasure(ABC):
    @abstractmethod
    def measure(self, a: Series, b: Series) -> float:
        pass


# class MinkowskiDistanceMeasure(DistanceMeasure):
#     m: int
#
#     def __init__(self, m: int = 1):
#         self.m = m
#
#     def measure(self, a: Series, b: Series) -> float:
#         return sum(abs(ai - bi) ** self.m for ai, bi in zip(a, b)) ** (1 / self.m)


class ManhattanDistanceMeasure(DistanceMeasure):
    def measure(self, a: Series, b: Series) -> float:
        return (a - b).abs().sum()


class EuclideanDistanceMeasure(DistanceMeasure):
    def measure(self, a: Series, b: Series) -> float:
        return np.sqrt(((a - b) ** 2).sum())


# class KNearestStack:
#     k_nearest: int
#     distances = []
#     items = []
#
#     def __init__(self, k_nearest: int):
#         self.k_nearest = k_nearest
#
#     def add(self, item, distance: float):
#         self.items.append(item)
#         self.distances.append(distance)
#
#     def get_nearest(self):
#         return [item for distance, item in sorted(zip(self.distances, self.items))[:self.k_nearest]]


# TODO improve, use condensing and editing
@dataclass(frozen=True)
class KNN:
    distance_measure: DistanceMeasure
    k_nearest: int
    Y: Series
    X: DataFrame

    def classify(self, input_x: Series):
        def apply_distance(row_a, row_b, distance_measure: DistanceMeasure) -> float:
            return distance_measure.measure(row_a, row_b)

        result_distance = self.X.apply(apply_distance, axis=1, args=(input_x, self.distance_measure)).to_frame('dist')
        result_distance['class'] = self.Y

        nearest_classes = result_distance.nsmallest(self.k_nearest, 'dist').iloc[:, 1]
        return nearest_classes.mode()[0]
