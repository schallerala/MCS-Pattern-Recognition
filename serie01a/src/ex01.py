import pandas as pd
from pandas import DataFrame, Series
from tqdm import tqdm

from KNN import KNN, ManhattanDistanceMeasure, EuclideanDistanceMeasure

Ks = [1, 3, 5, 10, 15]
distance_measures = [ManhattanDistanceMeasure(), EuclideanDistanceMeasure()]


columns = ['class']
X_columns = [i for i in range(28 * 28)]
columns.extend(X_columns)


def load_dataframe(filename: str, nrows: int = 1000) -> DataFrame:
    df = pd.read_csv(filename, header=None, names=columns, nrows=nrows)
    print(df.info)

    # Y = df['class'].tolist()
    # X = df[X_columns]
    # return df, X, Y
    return df


YX = load_dataframe('../train.csv')
YX_test = load_dataframe('../test.csv', 100)

# https://stackoverflow.com/a/34365537/3771148
# to track the progress of apply
tqdm.pandas()


def classify_test(row: Series, model):
    y = row.iloc[0]
    y2 = model.classify(row.iloc[1:])
    return pd.Series([y, y2, y == y2], index=['correct_class', 'classified_class', 'correct_classification'])


for K in Ks:
    for measure in distance_measures:
        print(f"{K}: {measure}")
        classifier = KNN(measure, K, YX)
        results = YX_test.progress_apply(classify_test, axis=1, args=(classifier,))

        print(results['correct_classification'].value_counts())
