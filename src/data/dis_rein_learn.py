from load import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_neuron_data(df, windows=None):
    """
    Convert the raw neuron data into a feature vectors using defined windows.
    Since rats have a different number of neurons, only use one individual rat's data
    
    Returns:
    - Features: N x (num_neurons x num_windows) feature matrix
    - Labels: lever press (0  or 1) per trial
    """
    if windows is None:
        windows = [
            (0, 20),  # Pre-tone
            (20, 80),  # During tone
            (80, 100)  # Post-tone
        ]
    
    features = []
    for _, row in df.iterrows():
        trial_data = row[5:].values
        neurons = trial_data.reshape



def distributed_reinforcement_learning():
    """
    Ex
    """
    
    df = load_file('PFC_con_4.csv')