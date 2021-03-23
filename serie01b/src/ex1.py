
import multiprocessing

import numpy as np
from math import ceil
from joblib import Parallel, delayed

from KMeans import KMeans
from loading_data import load_dataframe

Ks = [5, 7, 9, 10, 12, 15]

TOTAL_TRAIN_LINES = 27000

TEST_LINES_PER_BATCH = 25000

diff = TOTAL_TRAIN_LINES - TEST_LINES_PER_BATCH

starts = np.arange(0, diff, diff / len(Ks))

# (start, num_lines, K)
process_params = [(int(s), e, K) for s, e, K in zip(starts, [TEST_LINES_PER_BATCH] * len(Ks), Ks)]

def batch_process(start, num_lines, K):
    _, X = load_dataframe('../train.csv', nrows=num_lines, skip=start)
    print("loaded")
    model = KMeans(K).fit(X)
    print("fitted")
    return (K, model.c_index_score(), model.dunn_index_score())


if __name__ == '__main__':
    r = Parallel(n_jobs=multiprocessing.cpu_count(), verbose=10)(delayed(batch_process)(*args) for args in process_params)
    print(('K', 'c-index', 'dunn-index'))
    for result in r:
        print(result)
