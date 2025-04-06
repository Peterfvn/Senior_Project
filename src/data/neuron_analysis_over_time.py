from load import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def preparedata():
    """
    Load and prepare the data. We are analyzing each neuron's activity over time for different trials.
    """
    df = load_file('PFC_con_4.csv') # This is all I'm doing I suppose
    return df

def heatmap_single_neuron(df):
    """
    Plot a heatmap of a single neuron's activity over time.
    """

    # Create the heatmap
    vmax = np.percentile(df.iloc[:, 5:], 99)
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(df.iloc[:, 5:], cmap='viridis', cbar=True, yticklabels=False, xticklabels=False, vmin=-30, vmax=vmax)

    # Add a dividing line between positive and negative trials
    ax.axhline(50, color='black', linewidth=2, linestyle='--')

    # Set labels and title
    plt.xlabel("Time")
    plt.ylabel("Trials")
    plt.title(f"Neuron {df.columns[1]}")
    plt.savefig(f"heatmap_neuron_{df.columns[1]}.png")

def main():
    df = preparedata()
    rats = df[df.columns[0]].unique()

    # Just use rat 1 for now
    df_rat = df[df[df.columns[0]] == 1.0]
    
    # Filter down to just one neuron
    neuron_num = 1.0
    df_neuron = df_rat[df_rat[df_rat.columns[1]] == neuron_num]

    # Sort the neuron by trial type to divide the heatmap
    df_neuron = df_neuron.sort_values(by=[df_neuron.columns[3]])
    
    # Plot the heatmap
    heatmap_single_neuron(df_neuron)



main()