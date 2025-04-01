from data.load import *
import numpy as np
import pandas as pd

def calculate_press_counts(df):
    """Calculate how many times each rat pressed for its positive and negative tone trials"""
    results = []

    rat_ids = df[df.columns[0]].unique()

    for rat_id in rat_ids:
        rat_df = df[df[df.columns[0]] == rat_id]

        ds_plus_data = rat_df[rat_df[df.columns[3]] == 1]
        ds_minus_data = rat_df[rat_df[df.columns[3]] == 0]
        ds_plus_presses = ds_plus_data[df.columns[4]].sum()
        ds_minus_presses = ds_minus_data[df.columns[4]].sum()

        results.append({'rat_id': rat_id, 'positive_presses': ds_plus_presses, 'negative_presses': ds_minus_presses})

    return pd.DataFrame(results)

def main():
    df = load_file('PFC_con_4.csv')
    df2 = load_file('PFC_con_5.csv')
    df = trial_avg(df)
    df2 = trial_avg(df2)

    results = calculate_press_counts(df)
    results2 = calculate_press_counts(df2)

    print(results)
    print(results2)
    

main()