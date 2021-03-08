from abc import abstractmethod, ABC
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.base import ClassifierMixin, BaseEstimator

from tqdm import tqdm
# # https://stackoverflow.com/a/34365537/3771148
# # to track the progress of apply
tqdm.pandas()


columns = ['class']
X_columns = [i for i in range(28 * 28)]
columns.extend(X_columns)


def load_dataframe(filename: str, nrows: int = 1000, skip: int = 0) -> (Series, DataFrame):
    df = pd.read_csv(filename, header=None, names=columns, nrows=nrows, skiprows=skip)
    # print(df.info)

    return df['class'], df.iloc[:, 1:]



class DistanceMeasure(ABC):
    def apply(self, a: DataFrame, b: DataFrame) -> DataFrame:
        def distance(row: Series, df: DataFrame, measure_fn):
            return measure_fn(row, df)

        # if a.shape < b.shape:
        #     smallest = a
        #     biggest = b
        # else:
        #     smallest = b
        #     biggest = a

        return a.progress_apply(distance, axis=1, args=(b, self.measure))

    @abstractmethod
    def measure(self, a: Series, b: DataFrame) -> float:
        pass

    def __repr__(self):
        return self.__class__.__name__


# class MinkowskiDistanceMeasure(DistanceMeasure):
#     m: int
#
#     def __init__(self, m: int = 1):
#         self.m = m
#
#     def measure(self, a: Series, b: Series) -> float:
#         return sum(abs(ai - bi) ** self.m for ai, bi in zip(a, b)) ** (1 / self.m)


class ManhattanDistanceMeasure(DistanceMeasure):
    def measure(self, a: Series, b: DataFrame) -> float:
        return (a - b).abs().sum(1)


class EuclideanDistanceMeasure(DistanceMeasure):

    def measure(self, a: Series, b: DataFrame) -> float:
        return np.sqrt(((a - b) ** 2).sum(1))


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



# def prepare_masks(new_shape: (int, int)) -> []:
#     x, y = new_shape
#     x_parts = (int)(28 / x)
#     y_parts = (int)(28 / y)
#
#     mask = np.ones(new_shape, dtype=np.int0)
#
#     masks = []
#
#     for xi in range(x_parts):
#         x_start = x * xi
#         x_end = x_start + x
#         for yi in range(y_parts):
#             y_start = y * yi
#             y_end = y_start + y
#
#             pad_x_before = x_start
#             pad_x_after = 28 - x_end
#             pad_y_before = y_start
#             pad_y_after = 28 - y_end
#
#             mask_i = np.pad(mask, ((pad_x_before, pad_x_after), (pad_y_before, pad_y_after)))
#
#             masks.append(mask_i.flatten() == 1)
#
#     return masks
#
#
#
# def reshape_frame(df: DataFrame, masks: [[]]) -> DataFrame:
#     sums = [df.iloc[:, mask].sum(1) for mask in masks]
#     return pd.DataFrame(np.array(sums).T)




class KNN(ClassifierMixin, BaseEstimator):
    distance_measure: DistanceMeasure
    k_nearest: int
    X: DataFrame
    Y: Series

    def __init__(self, distance_measure: DistanceMeasure, k_nearest: int):
        self.distance_measure = distance_measure
        self.k_nearest = k_nearest

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        return self

    def predict(self, x: DataFrame):
        # Each col is the result x_i with the distance compared to train X (rows)
        result_distance = self.distance_measure.apply(self.X, x)

        predictions = []

        # for each col, distance of x_i with the train set
        # select the k smallest and take the most present class
        for col in result_distance.columns:
            nearest = result_distance.nsmallest(self.k_nearest, col).index
            nearest_classes = self.Y.iloc[nearest]
            predictions.append(nearest_classes.mode()[0])

        return pd.Series(predictions)
