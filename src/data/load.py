import pandas as pd
import os
import numpy as np
import torch
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

def load_data():
    # All filespaths
    load_dotenv()
    data_path = os.getenv("NEURON_DATA_DIR")
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    # Load all csv files into a list of tuples (dataframe, filename)
    dfs = [(pd.read_csv(os.path.join(data_path, f)), f) for f in csv_files]

    return dfs

def load_file(filename):
    load_dotenv()
    data_path = os.getenv("NEURON_DATA_DIR")

    if not os.path.exists(os.path.join(data_path, filename)):
        raise FileNotFoundError(f"File {filename} not found in {data_path}")
    df = pd.read_csv(os.path.join(data_path, filename))
    return df


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

def prepareDataLoader(dataset, labels, batchsize=32):
    train, test, train_labels, test_labels = train_test_split(dataset, labels, test_size=0.2, random_state=42)


    train_dataset = TensorDataset(train, train_labels)
    test_dataset = TensorDataset(test, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

    return (train_dataloader, test_dataloader)

def clean_data(df):
    # There shouldn't be any NA rows, but in case
    df = df.dropna()

    # Convert all to numbers
    df = df.apply(pd.to_numeric, errors='coerce')

    # Filter rows of only 0s
    data_cols = df.columns[5:]
    df = df[(df[data_cols] != 0).any(axis=1)]
    
    return df

def population_avg(df):
    # Averages rats with same number and trial type. Possibly same outcome?
    population_df = df.groupby([df.columns[0], df.columns[3]]).mean().reset_index()
    return population_df
    

# Entry point for testing
if __name__ == "__main__":
    dfs = load_data()
    print(type(dfs[0][0]))

    df = dfs[0][0]
    clean_data(df)
    # Test tensorize_data
    print(tensorize_data(df, 1)[0])
