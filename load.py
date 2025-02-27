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

def tensorize_data(df, labels_num):
    # Assumes df is read directly from Neuron Data CSV

    # Drop rat number, cell number, trial number. Reindex
    df = df.drop(columns=df.columns[:3])
    df.columns = range(len(df.columns))

    # 1 = Press, 0 = DS+/DS-
    if labels_num != 1 and labels_num != 0:
        raise ValueError("labels must be either 0 or 1")

    # Extract labels
    labels = df[labels_num]
    labels = torch.tensor(labels, dtype=torch.long)

    # Drop both Press and DS+/DS- columns
    df = df.drop(columns=df.columns[:2])
    df = torch.tensor(df.to_numpy(), dtype=torch.float32)

    return (df, labels)

def clean_data(df):
    # There shouldn't be any NA rows, but in case
    df = df.dropna()

    # Convert all to numbers
    df = df.apply(pd.to_numeric, errors='coerce')

    # Filter rows of only 0s
    data_cols = df.columns[5:]
    df = df[(df[data_cols] != 0).any(axis=1)]
    
    return df

# Entry point for testing
if __name__ == "__main__":
    dfs = load_data()
    print(type(dfs[0][0]))

    df = dfs[0][0]
    clean_data(df)
    # Test tensorize_data
    print(tensorize_data(df, 1)[0])
