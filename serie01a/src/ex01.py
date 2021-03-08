import logging
from threading import Thread

from KNN import *
import itertools


Y, X = load_dataframe('../train.csv', nrows=2500)

from sklearn.model_selection import RandomizedSearchCV

dimensions = [14,7,4,2]
shapes = list(itertools.product(dimensions, dimensions))

hyperparameters = dict(reshape=shapes)
model = KNN(EuclideanDistanceMeasure(), 3, (28, 28))
rscv = RandomizedSearchCV(model, hyperparameters, n_jobs=4, random_state=42)
rscv.fit(X, Y)

best_params = rscv.best_params_
print(f"Best parameters: {best_params}")


model.set_params(**best_params)


print("Start with test set")



def thread_batch_job(skip, rows, model):
    y_test, x_test = load_dataframe('../test.csv', nrows=rows, skip=skip)
    # score = model.score(x_test, y_test)
    score = 0
    print(f"{score}: {model} [skip: {skip}]")


def thread_job(model):
    # total lines 15000
    for skip in range((int)(15000 / 1000)): # batches of 1000
        thread_batch_job(skip, 1000, model)


Ks = [1, 3, 5, 10, 15]
distance_measures = [ManhattanDistanceMeasure(), EuclideanDistanceMeasure()]


threads = []


Y, X = load_dataframe('../train.csv', nrows=1000)

for [K, dist] in itertools.product(Ks, distance_measures):
    thread = Thread(target=thread_job, args=(KNN(dist, K, best_params['reshape']).fit(X, Y)))
    threads.append(thread)
    thread.start()


for index, thread in enumerate(threads):
    logging.info("Main    : before joining thread %d.", index)
    thread.join()
    logging.info("Main    : thread %d done", index)


print("Done")
