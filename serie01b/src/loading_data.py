import pandas as pd
from pandas import DataFrame, Series
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
