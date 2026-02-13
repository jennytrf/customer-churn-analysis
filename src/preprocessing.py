import pandas as pd

def load_and_prepare_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["profit"] = df["revenue"] - df["cost"]
    return df
