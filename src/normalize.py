import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

"""This script didn't get used much. It explored normalization techniques I didn't end up using."""

def trial_avg(df):
    """Calculate population average by trial"""
    # Group by rat_id and trial_id, then calculate the mean for each group
    df_columns = df.columns.tolist()
    df = df.groupby([df.columns[0], df.columns[2]]).mean().reset_index()

    # Reset indexes
    df = df[df_columns]
    return df

def plot_trial_averages(df, name):
    """Plot trial-averaged population response"""
    load_dotenv()
    save_dir = os.getenv('VISUALIZATION_DIR')

    file = os.path.join(save_dir, name)
    
    num_rats = df[df.columns[0]].unique()
    fig, axes = plt.subplots(len(num_rats), 2, figsize=(15, 5 * len(num_rats)))
    fig.suptitle('Trial-Averaged Population Response')
    
    for idx, rat in enumerate(num_rats):
        rat_data = df[df[df.columns[0]] == rat]
        time = range(len(rat_data.columns[5:]))

        y_min = rat_data.iloc[:, 5:].min().min()
        y_max = rat_data.iloc[:, 5:].max().max()

        y_padding = (y_max - y_min) * 0.1
        y_min -= y_padding
        y_max += y_padding
        
        # Split by DS+ and DS-
        ds_plus = rat_data[rat_data[rat_data.columns[3]] == 1]
        ds_minus = rat_data[rat_data[rat_data.columns[3]] == 0]

        # Plot trials separated by DS+ and DS-
        for _, row in ds_plus.iterrows():
            axes[idx][0].plot(time, row[5:], alpha=0.1, color='gray')
            axes[idx][0].set_ylim(y_min, y_max)

        for _, row in ds_minus.iterrows():
            axes[idx][1].plot(time, row[5:], alpha=0.1, color='gray')
            axes[idx][1].set_ylim(y_min, y_max)

        axes[idx][0].set_xlabel('Time (ms)')
        axes[idx][0].set_ylabel('Activity')
        axes[idx][0].set_title(f'Rat {rat} DS+')

        axes[idx][1].set_xlabel('Time (ms)')
        axes[idx][1].set_ylabel('Activity')
        axes[idx][1].set_title(f'Rat {rat} DS-')
    
    plt.tight_layout()
    plt.savefig(file)
    plt.close()
    return

if __name__ == '__main__':
    load_dotenv()
    data_path = os.getenv("NEURON_DATA_DIR")
    df = pd.read_csv(os.path.join(data_path, "PFC_con_4.csv"))
    trial_df = trial_avg(df)

    