# Loads all my data into a list of pandas dataframes
import pandas as pd
import os
import numpy as np

def load_data():
    # All filespaths
    csv_files = [f for f in os.listdir("./Neuron Data")]
    # Load all csv files into a list of tuples (dataframe, filename)
    dfs = [(pd.read_csv(os.path.join("./Neuron Data", f)), f) for f in csv_files]

    return dfs

# Entry point for testing
if __name__ == "__main__":
    dfs = load_data()
    print(type(dfs[0][1]))