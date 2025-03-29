import pandas as pd
import os
import numpy as np
import torch
from dotenv import load_dotenv
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tslearn.metrics import dtw
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

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

# Not sure this is necessary considering my data is already in Z-Scores
def normalize_by_rat(df):
    rats = df[df.columns[0]].unique()
    normalized_df = pd.DataFrame()

    for rat in rats:
        rat_data = df[df[df.columns[0]] == rat]
        rat_data[:, 5:] = (rat_data[:, 5:] - rat_data[:, 5:].min()) / (rat_data.iloc[:, 5:].std())
        normalized_df = pd.concat([normalized_df, rat_data], ignore_index=True)

        return normalized_df
    
def cluster_rats(df, n_clusters=2):
    from sklearn.cluster import KMeans

    rat_features = df.groupby(df.columns[0]).agg({col: ['mean', 'max', 'std'] for col in df.columns[5:]}).mean(axis=1)

    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(rat_features.values.reshape(-1, 1))

    return dict(zip(rat_features.index, clusters))

def extract_regional_features(df, id):
    # Extract features on a by rat basis
    rat_data = df[df[df.columns[0]] == id][df.columns[5:]]
    rat_data = rat_data.values.flatten()
    
    # Assuming first 20/80/100 bins are divisors
    pre_tone = rat_data[:20]
    during_tone = rat_data[20:80] 
    lever_intro = rat_data[80:]
    
    features = {
        # Pre-tone features
        'pre_tone_mean': np.mean(pre_tone),
        'pre_tone_std': np.std(pre_tone),
        
        # During-tone features
        'tone_mean': np.mean(during_tone),
        'tone_std': np.std(during_tone),
        'tone_max': np.max(during_tone),
        'tone_min': np.min(during_tone),
        'tone_range': np.max(during_tone) - np.min(during_tone),
        
        # Response timing features
        'time_to_max': np.argmax(during_tone),
        'time_to_min': np.argmin(during_tone),
        
        # Lever introduction features
        'lever_mean': np.mean(lever_intro),
        'lever_response': np.mean(lever_intro) - np.mean(pre_tone)
    }
    
    return features, {'pre_tone': pre_tone, 'during_tone': during_tone, 'lever_intro': lever_intro}

def dtw_calculate_metrics(df):
    time_data = df.iloc[:, 5:].values
    n = len(time_data)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            distance = dtw(time_data[i], time_data[j])
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance

    return dist_matrix

def cluster_with_regional_dtw(df, n_clusters=2, region='during_tone'):
    """Cluster rats based on DTW distances for a specific temporal region."""
    rat_ids = df[df.columns[0]].unique()
    all_features = []
    time_series_by_region = {region: [] for region in ['pre_tone', 'during_tone', 'lever_intro']}
    rat_id_to_index = {}
    
    # Extract features and time series for each rat
    for i, rat_id in enumerate(rat_ids):
        features, regions = extract_regional_features(df, rat_id)
        all_features.append(features)
        
        for region_name, region_data in regions.items():
            time_series_by_region[region_name].append(region_data)
        
        rat_id_to_index[rat_id] = i
    
    # Calculate DTW distance matrix for the specified region
    region_time_series = time_series_by_region[region]
    print(region_time_series)
    exit()
    dtw_matrix = dtw_distance_matrix(region_time_series)
    
    # Apply hierarchical clustering using the DTW distance matrix
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, 
        affinity='precomputed', 
        linkage='average'
    )
    
    cluster_labels = clustering.fit_predict(dtw_matrix)
    
    # Create a dictionary mapping rat IDs to cluster labels
    clusters = {rat_id: cluster_labels[rat_id_to_index[rat_id]] for rat_id in rat_ids}
    
    # Create a feature DataFrame for further analysis
    feature_df = pd.DataFrame(all_features)
    feature_df['rat_id'] = rat_ids
    feature_df['cluster'] = cluster_labels
    
    return clusters, feature_df, dtw_matrix

def trial_avg(df):
    """Calculate population average by trial"""
    # Group by rat_id and trial_id, then calculate the mean for each group
    df_columns = df.columns.tolist()
    df = df.groupby([df.columns[0], df.columns[2]]).mean().reset_index()

    # Reset indexes
    df = df[df_columns]
    return df

# Entry point for testing
if __name__ == "__main__":
    dfs = load_data()
    print(type(dfs[0][0]))

    df = dfs[0][0]
    clean_data(df)
    # Test tensorize_data
    print(tensorize_data(df, 1)[0])
