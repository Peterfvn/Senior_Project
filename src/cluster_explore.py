from data.load import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import umap.umap_ as umap
import seaborn as sns
from collections import defaultdict

def prepare_data():
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

    rat_to_cluster = dict(zip(responses['Rat'], responses['Cluster']))
    df['Cluster'] = df[df.columns[0]].map(rat_to_cluster)

    return df

def neuron_per_trial():
    """
    Create a feature vector with neuron data for each trial.
    [mean_n1, mean_n2, ..., mean_nn, std_n1, std_n2, ..., std_nn]
    This loses temporal data. Will be improved later.
    """
    from sklearn.metrics import accuracy_score, confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = load_file('PFC_con_4.csv')

    time_cols = df.columns[5:]

    # Unique trial ID
    df['Trial Key'] = list(zip(df[df.columns[0]], df[df.columns[2]]))

    rat_features = defaultdict(list)
    rat_labels = defaultdict(list)
    rat_cue = defaultdict(list)

    for trial_key, group in df.groupby([df.columns[0], 'Trial Key']):
        rat_id = group[df.columns[0]].iloc[0]
        press = group[df.columns[4]].iloc[0]
        cue_type = group[df.columns[3]].iloc[0]

        mean_vec = group[time_cols].mean(axis=1).values
        std_vec = group[time_cols].std(axis=1).values
        trial_vec = np.concatenate([mean_vec, std_vec])

        rat_features[rat_id].append(trial_vec)
        rat_labels[rat_id].append(press)
        rat_cue[rat_id].append(cue_type)

    for rat_id in rat_features:
        X = np.stack(rat_features[rat_id])
        y = np.array(rat_labels[rat_id])
        cues = np.array(rat_cue[rat_id])

        # Filter negatives (maybe or maybe not do this)
        neg_mask = cues == 0
        X_neg = X[neg_mask]
        y_neg = y[neg_mask]

        print(f"Training model for Rat {rat_id}")

        X_train, X_test, y_train, y_test = train_test_split(X_neg, y_neg, test_size=0.2, stratify=y_neg, random_state=42)
        clf = LogisticRegression(max_iter=1000)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f"  ✅ Accuracy: {acc:.3f}")
        print("  Confusion matrix:")
        print(cm)

def plot_umap(df):
    tones = df.iloc[:, 5:105].values
    during_tone = tones[:, :80]

    umap_embedding = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='euclidean', random_state=42)
    embeddings = umap_embedding.fit_transform(during_tone)
    labels = df[df.columns[105]]

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="coolwarm", alpha=0.8)

    unique_labels = np.unique(labels)
    for label in unique_labels:
        plt.scatter([], [], color=scatter.cmap(scatter.norm(label)), label=label)
    plt.legend(title="Labels")
    plt.title('UMAP Visualization of Rat Data')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.savefig('umap_visualization_pretone_tone.png')


def main():
    neuron_per_trial()



main()