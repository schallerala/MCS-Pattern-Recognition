import pandas as pd
from pandas import DataFrame, Series
from tqdm import tqdm

from KNN import KNN, ManhattanDistanceMeasure, EuclideanDistanceMeasure

Ks = [3, 5]
# Ks = [1, 3, 5, 10, 15]
distance_measures = [ManhattanDistanceMeasure(), EuclideanDistanceMeasure()]


columns = ['class']
X_columns = [i for i in range(28 * 28)]
columns.extend(X_columns)


def load_dataframe(filename: str, nrows: int = 1000) -> (Series, DataFrame):
    df = pd.read_csv(filename, header=None, names=columns, nrows=nrows)
    print(df.info)

    return df['class'], df.iloc[:, 1:]


Y, X = load_dataframe('../train.csv')
Y_test, X_test = load_dataframe('../test.csv', 100)

# https://stackoverflow.com/a/34365537/3771148
# to track the progress of apply
tqdm.pandas()


def classify_test(row: Series, model):
    return model.classify(row)


for K in Ks:
    for measure in distance_measures:
        print(f"{K}: {measure}")
        classifier = KNN(measure, K, Y, X)
        classified_as = X_test.progress_apply(classify_test, axis=1, args=(classifier,))

        print((classified_as == Y_test).value_counts())
