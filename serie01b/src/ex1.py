
import multiprocessing

import numpy as np
from math import ceil
from joblib import Parallel, delayed

from KMeans import KMeans
from loading_data import load_dataframe

Ks = [5, 7, 9, 10, 12, 15]

TOTAL_TEST_LINES = 15000

test_lines_per_batch = ceil(TOTAL_TEST_LINES / len(Ks))
np.arange(0, TOTAL_TEST_LINES, test_lines_per_batch)
starts = np.arange(0, TOTAL_TEST_LINES, test_lines_per_batch)
num_lines = [test_lines_per_batch] * len(starts)


# (start, num_lines, K)
process_params = [(s, e, K) for s, e, K in zip(starts, num_lines, Ks)]



def batch_process(start, num_lines, K):
    _, X = load_dataframe('../train.csv', nrows=None)
    model = KMeans(K).fit(X)
    return (K, model.c_index_score(), model.dunn_index_score())


if __name__ == '__main__':
    r = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(batch_process)(*args) for args in process_params)
    print(('K', 'c-index', 'dunn-index'))
    for result in r:
        print(result)
