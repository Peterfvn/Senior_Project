from data.load import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import umap.umap_ as umap
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
    [mean_pre_n1, std_pre_n1, ..., mean_post_nn, std_post_nn]
    This loses temporal data. Will be improved later.
    """
    df = load_file('PFC_con_4.csv')

    time_cols = df.columns[5:]
    pre_cols = time_cols[:20]
    during_cols = time_cols[20:80]
    post_cols = time_cols[80:]

    # Unique trial ID
    df['Trial Key'] = list(zip(df[df.columns[0]], df[df.columns[2]]))

    rat_features = defaultdict(list)
    rat_labels = defaultdict(list)
    rat_cue = defaultdict(list)
    rat_models = {}
    rat_feature_names = defaultdict(list)

    for trial_key, group in df.groupby([df.columns[0], 'Trial Key']):
        rat_id = group[df.columns[0]].iloc[0]
        press = group[df.columns[4]].iloc[0]
        cue_type = group[df.columns[3]].iloc[0]
 
        pre_mean = group[pre_cols].mean(axis=1).values 
        during_mean = group[during_cols].mean(axis=1).values 
        post_mean = group[post_cols].mean(axis=1).values
 
        pre_std = group[pre_cols].std(axis=1).values 
        during_std = group[during_cols].std(axis=1).values 
        post_std = group[post_cols].std(axis=1).values
 
        trial_vec = np.concatenate([pre_mean, pre_std, during_mean, during_std, post_mean, post_std])

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

        # Make feature names
        num_units = X.shape[1] // 6
        feature_names = []
        for i in range(num_units):
            feature_names += [f"pre_mean_{i}", f"pre_std_{i}", f"during_mean_{i}", f"during_std_{i}", f"post_mean_{i}", f"post_std_{i}"]

        rat_feature_names[rat_id] = feature_names

        print(f"Training model for Rat {rat_id}")

        X_train, X_test, y_train, y_test = train_test_split(X_neg, y_neg, test_size=0.2, stratify=y_neg, random_state=42)

        # Oversampling
        from sklearn.utils import resample
        X_press = X_train[y_train == 1]
        X_nopress = X_train[y_train == 0]

        if len(X_press) == 0 or len(X_nopress) == 0:
            print("Skipping Rat {rat_id} due to insufficient data.")
            continue
    
        X_press_unsampled, y_press_unsampled = resample(X_press, y_train[y_train==1], replace=True, n_samples=len(X_nopress), random_state=42)

        X_train = np.vstack([X_nopress, X_press_unsampled])
        y_train = np.hstack([y_train[y_train==0], y_press_unsampled])

        # clf = LogisticRegression(max_iter=1000)
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        # print(f"  ✅ Accuracy: {acc:.3f}")
        # print(classification_report(
        #     y_test,
        #     y_pred,
        #     labels=[0, 1],  # Expect both classes
        #     target_names=['No Press', 'Press'],
        #     zero_division=0  # Avoid division errors when precision/recall is undefined
        # ))

        # Save the model
        rat_models[rat_id] = clf

    """Cross model training"""
    X = np.stack(rat_features[10.0])
    y = np.array(rat_labels[10.0])
    cues = np.array(rat_cue[10.0])

    X_test = np.stack(rat_features[15.0])
    y_test = np.array(rat_labels[15.0])

    # Filter negatives
    neg_mask = cues == 0
    X_neg = X[neg_mask]
    y_neg = y[neg_mask]

    X_test_neg = X_test[neg_mask]
    y_test_neg = y_test[neg_mask]

    # X_press = X_neg[y_neg == 1]
    # X_nopress = X_neg[y_neg == 0]

    # X_press_unsampled, y_press_unsampled = resample(X_press, y_neg[y_neg==1], replace=True, n_samples=len(X_nopress), random_state=42)

    # X_train = np.vstack([X_nopress, X_press_unsampled])
    # y_train = np.hstack([y_train[y_train==0], y_press_unsampled])

    # print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    # exit()

    clf = LogisticRegression(max_iter=1000)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    exit()
    clf.fit(X_neg, y_neg)

    y_pred = clf.predict(X_test_neg)
    acc = accuracy_score(y_test_neg, y_pred)

    print(f"\n📊 Cross-Rat Transfer: Train on Rat {10}, Test on Rat {15}")
    print(classification_report(y_test, y_pred, target_names=["No Press", "Press"]))

def cross_model_training():
    """Train a model on one rat's data and test on another rat's data"""
    df = load_file("PFC_con_4.csv")
    df = clean_data(df)
    df = trial_avg(df)
    rats = df[df.columns[0]].unique()

    # Using trial data instead of individual neuron data

    time_cols = df.columns[5:105]
    pre_tone = time_cols[:20]
    during_tone = time_cols[20:80]
    post_tone = time_cols[80:]

    rat_features = defaultdict(list)
    rat_labels = defaultdict(list)
    rat_cue = defaultdict(list)
    
    for rat in rats:
        rat_df = df[df[df.columns[0]] == rat]
        all_features = []
        for index, row in rat_df.iterrows():
            pre_mean = row.iloc[pre_tone].mean()
            during_mean = row.iloc[during_tone].mean()
            post_mean = row.iloc[post_tone].mean()

            pre_std = row.iloc[pre_tone].std()
            during_std = row.iloc[during_tone].std()
            post_std = row.iloc[post_tone].std()

            trial_vec = np.array([pre_mean, pre_std, during_mean, during_std, post_mean, post_std])
            all_features.append(trial_vec)

            press_label = row[df.columns[4]]
            cue_label = row[df.columns[3]]

            rat_labels[rat].append(press_label)
            rat_cue[rat].append(cue_label)
        rat_features[rat] = all_features

    # Train a model on one rat's data and test on another rat's data
    cls = LogisticRegression(max_iter=1000)
    X = np.stack(rat_features[15.0])
    y = np.array(rat_labels[15.0])
    
    X_test = np.stack(rat_features[3.0])
    y_test = np.array(rat_labels[3.0])

    cls.fit(X, y)
    y_pred = cls.predict(X_test)
    print(f"\n📊 Cross-Rat Transfer: Train on Rat {15}, Test on Rat {10}")
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred, target_names=["No Press", "Press"]))
    

    


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
    cross_model_training()



main()