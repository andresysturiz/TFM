import pandas as pd

def load_dataset(path, target_col):
    df = pd.read_csv(path)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    return df, X, y
