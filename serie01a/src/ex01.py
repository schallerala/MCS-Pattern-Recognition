# Y, X = load_dataframe('../train.csv', nrows=2500)
#
# from sklearn.model_selection import RandomizedSearchCV
#
# dimensions = [14,7,4,2]
# shapes = list(itertools.product(dimensions, dimensions))
#
# hyperparameters = dict(reshape=shapes)
# model = KNN(EuclideanDistanceMeasure(), 3, (28, 28))
# rscv = RandomizedSearchCV(model, hyperparameters, n_jobs=4, random_state=42)
# rscv.fit(X, Y)
#
# best_params = rscv.best_params_
# print(f"Best parameters: {best_params}")
#
#
# model.set_params(**best_params)


import itertools
import multiprocessing

import numpy as np
from math import ceil
from joblib import Parallel, delayed

from KNN import load_dataframe, KNN, EuclideanDistanceMeasure, ManhattanDistanceMeasure

Ks = [1, 3, 5, 10, 15]
distance_measures = [ManhattanDistanceMeasure(), EuclideanDistanceMeasure()]


TOTAL_LINES = 15000


combinations = list(itertools.product(Ks, distance_measures))
combinations_count = len(combinations)


test_lines_per_batch = ceil(TOTAL_LINES / combinations_count)
np.arange(0, TOTAL_LINES, test_lines_per_batch)
starts = np.arange(0, TOTAL_LINES, test_lines_per_batch)
num_lines = [test_lines_per_batch] * len(starts)

# (start, num_lines, K, distance)
process_params = [(s, e, K, d) for s, e, (K, d) in zip(starts, num_lines, combinations)]


def batch_process(start, num_lines, K, distance):
    Y, X = load_dataframe('../train.csv', nrows=None)
    model = KNN(distance, K).fit(X, Y)
    y_test, x_test = load_dataframe('../test.csv', nrows=num_lines, skip=start)
    score = model.score(x_test, y_test)
    return (score, K, distance)

if __name__ == '__main__':
    r = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(batch_process)(*args) for args in process_params)
    print(r)
