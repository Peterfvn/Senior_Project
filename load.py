# Loads all my data into a list of pandas dataframes
import pandas as pd
import os
import numpy as np
import torch

def load_data():
    # All filespaths
    csv_files = [f for f in os.listdir("./Neuron Data")]
    # Load all csv files into a list of tuples (dataframe, filename)
    dfs = [(pd.read_csv(os.path.join("./Neuron Data", f)), f) for f in csv_files]

    return dfs

def tensorize_data(df, col_split):
    # Assumes df is read directly from Neuron Data CSV

    # Drop rat number, cell number, trial number. Reindex
    df = df.drop(columns=df.columns[:3])
    df.columns = range(len(df.columns))

    # 1 = Press, 0 = DS+/DS-
    if col_split != 1 and col_split != 0:
        raise ValueError("col_split must be either 0 or 1")

    # Extract labels
    labels = df[col_split]
    labels = torch.tensor(labels, dtype=torch.float32)

    # Drop both Press and DS+/DS- columns
    df = df.drop(columns=df.columns[:2])
    df = torch.tensor(df.to_numpy(), dtype=torch.float32)

    return (df, labels)

# Entry point for testing
if __name__ == "__main__":
    dfs = load_data()
    print(type(dfs[0][0]))

    # Test tensorize_data
    df = dfs[0][0]
    print(tensorize_data(df, 1)[0])