import pandas as pd
import numpy as np
from src.preprocessing import Preprocess

def test_drop_features():

    p = Preprocess()

    d = {
        'A': [1, 2, 3, 4, 5],
        'B': [3, 2, 3, 2.3, 32.3],
        'C': [2.3, 3.2, 4.2, 5.2, 3.2],
        'D': ['Worcester', 'Boston', 'Chicago', 'New York', 'Irvine']
    }

    df = pd.DataFrame(d)

    df = p.drop_features(df, ['A', 'C'])

    assert 'A' not in df.columns and 'C' not in df.columns, "Preprocess Fail: Dropping unimportant features failed"
    assert 'B' in df.columns and 'D' in df.columns, "Preprocess Fail: Dropping unimportant features failed"

def test_keep_features():

    p = Preprocess()

    d = {
        'A': [1, 2, 3, 4, 5],
        'B': [3, 2, 3, 2.3, 32.3],
        'C': [2.3, 3.2, 4.2, 5.2, 3.2],
        'D': ['Worcester', 'Boston', 'Chicago', 'New York', 'Irvine']
    }

    df = pd.DataFrame(d)

    df = p.keep_features(df, ['B', 'D'])

    assert 'A' not in df.columns, "Preprocess Fail: Keeping only required features failed"
    assert 'C' not in df.columns, "Preprocess Fail: Keeping only required features failed"
    assert 'B' in df.columns, "Preprocess Fail: Keeping only required features failed"
    assert 'D' in df.columns, "Preprocess Fail: Keeping only required features failed"

def test_clean_column_names():

    lst = ['$aak_weisk%', "_____askf%_23#"]

    p = Preprocess()

    result = [p.clean_column(col) for col in lst]

    assert result == ["aak_weisk", "askf_23"], "Preprocess Fail: Renaming column names for easier processing failed."


def test_replace_missing_values():

    p = Preprocess()

    d = {
        'A': [1, 2, 3, np.nan, 5],
        'B': [3, 2, 3, np.nan, 32.3],
        'C': [2.3, 3.2, 4.2, 5.2, np.nan]
    }

    df = pd.DataFrame(d)

    df = p.replace_missing_values(df)

    # Check if the nan is filled with median
    assert df.loc[3, 'B'] == 3, "Preprocess Fail: Replacing missing values with median failed"

    # check if all the nan values are filled
    assert df.isnull().sum().sum() == 0, "Preprocess Fail: Replacing missing values with median failed"

def test_scale_features():

    p = Preprocess()

    d = {
        'A': [100, 122, 23, 232, 78],
        'B': [0.3, 2.6, 300, 34, 32.3],
        'C': [2.3, 3.2, 4.2, 5.2, 678]
    }

    df = pd.DataFrame(d)

    df = p.scale_features(df)

    assert df.min().min() >= 0 and df.max().max() <= 1, "Preprocess Fail: Scaling features to between 0 and 1 failed."

def test_remove_features_with_zero_variance():

    p = Preprocess()

    d = {
        'A': [100, 122, 23, 232, 78],
        'B': [0.3, 2.6, 300, 34, 32.3],
        'C': [2.3, 3.2, 4.2, 5.2, 678],
        'D': [5, 5, 5, 5, 5]
    }

    df = pd.DataFrame(d)

    df = p.remove_features_with_zero_variance(df)

    assert not 'D' in df.columns, "Preprocess Fail: Removing features with zero variance failed."



