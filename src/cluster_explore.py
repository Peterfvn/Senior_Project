from data.load import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def main():
    df = load_file('PFC_con_4.csv')
    df = clean_data(df)
    df = trial_avg(df)

    rats = df[df.columns[0]].unique()
    responses = pd.DataFrame(columns=['Rat', 'Positive Rate', 'Negative Rate'])

    for rat in rats:
        df_rat = df[df[df.columns[0]] == rat]
        ds_plus = df_rat[df_rat[df_rat.columns[3]] == 1]
        ds_minus = df_rat[df_rat[df_rat.columns[3]] == 0]

        ds_plus_sum = ds_plus[df_rat.columns[4]].sum()
        ds_minus_sum = ds_minus[df_rat.columns[4]].sum()

        responses.loc[len(responses)] = [rat, ds_plus_sum/50, ds_minus_sum/50]
    
    kmeans = KMeans(n_clusters=4, random_state=0)
    responses['Cluster'] = kmeans.fit_predict(responses[['Positive Rate', 'Negative Rate']])

    plt.figure(figsize=(10, 6))
    for cluster in responses['Cluster'].unique():
        cluster_data = responses[responses['Cluster'] == cluster]
        plt.scatter(cluster_data['Positive Rate'], cluster_data['Negative Rate'], label=f'Cluster {cluster}')
        for _, row in cluster_data.iterrows():
            plt.text(row['Positive Rate'], row['Negative Rate'], str(row['Rat']), fontsize=8, ha='center', va='center')

    plt.xlabel('Positive Rate')
    plt.ylabel('Negative Rate')
    plt.title('Cluster Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig('cluster_plot.png')

    print(responses)


main()