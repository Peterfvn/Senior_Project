import numpy as np
import pandas as pd
from tslearn.metrics import dtw
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from data.load import *

def extract_regional_features(df, rat_id, time_cols):
    """Extract features from different temporal regions of neural data."""
    rat_data = df[df[df.columns[0]] == rat_id][time_cols].values.flatten()
    
    # Define region boundaries (in time bins)
    pre_tone = rat_data[:10]
    during_tone = rat_data[10:80]  # Assuming 70 bins for 3.5s
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

def dtw_distance_matrix(time_series_data):
    """Calculate pairwise DTW distances between time series."""
    n = len(time_series_data)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # Calculate DTW distance between time series i and j
            distance = dtw(time_series_data[i], time_series_data[j])
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
    
    return dist_matrix

def cluster_with_regional_dtw(df, time_cols, n_clusters=2, region='during_tone'):
    """Cluster rats based on DTW distances for a specific temporal region."""
    rat_ids = df[df.columns[0]].unique()
    all_features = []
    time_series_by_region = {region: [] for region in ['pre_tone', 'during_tone', 'lever_intro']}
    rat_id_to_index = {}
    
    # Extract features and time series for each rat
    for i, rat_id in enumerate(rat_ids):
        features, regions = extract_regional_features(df, rat_id, time_cols)
        all_features.append(features)
        
        for region_name, region_data in regions.items():
            time_series_by_region[region_name].append(region_data)
        
        rat_id_to_index[rat_id] = i
    
    # Calculate DTW distance matrix for the specified region
    region_time_series = time_series_by_region[region]
    dtw_matrix = dtw_distance_matrix(region_time_series)
    
    # Apply hierarchical clustering using the DTW distance matrix
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters, 
        metric='precomputed', 
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

def visualize_clusters(df, clusters, time_cols, region_start_idx=None, region_end_idx=None):
    """Visualize time series data grouped by clusters."""
    unique_clusters = set(clusters.values())
    
    if region_start_idx is None or region_end_idx is None:
        # Use all time columns if no specific region is specified
        region_time_cols = time_cols
    else:
        # Use only the columns for the specified region
        region_time_cols = time_cols[region_start_idx:region_end_idx]
    
    # Create a figure with subplots for each cluster
    fig, axes = plt.subplots(len(unique_clusters), 1, figsize=(12, 4*len(unique_clusters)))
    if len(unique_clusters) == 1:
        axes = [axes]
    
    for cluster_id in unique_clusters:
        # Get the rats in this cluster
        cluster_rats = [rat_id for rat_id, cluster in clusters.items() if cluster == cluster_id]
        
        ax = axes[cluster_id]
        ax.set_title(f'Cluster {cluster_id} Responses')
        
        # Plot each rat's time series in this cluster
        for rat_id in cluster_rats:
            rat_data = df[df[df.columns[0]] == rat_id][region_time_cols].values.flatten()
            ax.plot(rat_data, label=f'Rat {rat_id}')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Activity')
        ax.legend()
    
    plt.tight_layout()
    return fig

def visualize_dendrogram(dtw_matrix, rat_ids):
    """Visualize hierarchical clustering as a dendrogram."""
    # Convert distance matrix to condensed form for linkage
    condensed_dist = []
    for i in range(len(dtw_matrix)):
        for j in range(i+1, len(dtw_matrix)):
            condensed_dist.append(dtw_matrix[i, j])
    
    # Compute linkage matrix
    Z = linkage(condensed_dist, method='average')
    
    # Plot dendrogram
    plt.figure(figsize=(10, 6))
    dendrogram(Z, labels=[f'Rat {rat}' for rat in rat_ids])
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Rat ID')
    plt.ylabel('DTW Distance')
    plt.show()

def train_rnn_with_clusters(df, clusters, time_cols, target_col, test_size=0.2):
    """Train an RNN model using cluster information with PyTorch."""
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.model_selection import train_test_split
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # PyTorch LSTM Model class
    class LSTMModel(nn.Module):
        def __init__(self, input_size, output_size):
            super(LSTMModel, self).__init__()
            self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True, return_sequences=True)
            self.dropout1 = nn.Dropout(0.2)
            self.lstm2 = nn.LSTM(input_size=64, hidden_size=32, batch_first=True)
            self.dropout2 = nn.Dropout(0.2)
            self.fc1 = nn.Linear(32, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, output_size)
            self.sigmoid = nn.Sigmoid() if output_size == 1 else nn.Softmax(dim=1)
            
        def forward(self, x):
            # First LSTM layer
            x, _ = self.lstm1(x)
            x = self.dropout1(x)
            
            # Second LSTM layer
            x, _ = self.lstm2(x)
            x = self.dropout2(x)
            
            # Fully connected layers
            x = self.fc1(x[:, -1, :])  # Take the output from the last time step
            x = self.relu(x)
            x = self.fc2(x)
            
            # Output activation
            x = self.sigmoid(x)
            return x
    
    # Prepare data
    X = []
    y = []
    cluster_info = []
    
    for rat_id in df[df.columns[0]].unique():
        rat_data = df[df[df.columns[0]] == rat_id]
        rat_time_series = rat_data[time_cols].values
        rat_target = rat_data[target_col].values[0]  # Assuming target is the same for all rows of a rat
        rat_cluster = clusters.get(rat_id, 0)  # Default to cluster 0 if not found
        
        X.append(rat_time_series)
        y.append(rat_target)
        cluster_info.append(rat_cluster)
    
    X = np.array(X)
    y = np.array(y)
    cluster_info = np.array(cluster_info)
    
    print(f'X: {X}, y: {y}')
    exit()
    # Split data into train and test sets
    X_train, X_test, y_train, y_test, clusters_train, clusters_test = train_test_split(
        X, y, cluster_info, test_size=test_size, random_state=42, stratify=cluster_info
    )
    
    # Training parameters
    num_epochs = 100
    batch_size = 8
    patience = 10
    
    # Build RNN models with cluster-specific processing
    num_clusters = len(set(clusters.values()))
    models = []
    
    # Train separate models for each cluster
    for cluster_id in range(num_clusters):
        # Get data for this cluster
        cluster_mask = clusters_train == cluster_id
        if np.sum(cluster_mask) < 2:  # Skip if too few samples
            continue
        
        X_cluster = X_train[cluster_mask]
        y_cluster = y_train[cluster_mask]
        
        # Check if binary or multiclass classification
        is_binary = len(set(y_train)) <= 2
        output_size = 1 if is_binary else len(set(y_train))
        
        # Convert data to PyTorch tensors
        X_cluster_tensor = torch.FloatTensor(X_cluster)
        # For binary classification, ensure target is shaped properly
        if is_binary:
            y_cluster_tensor = torch.FloatTensor(y_cluster).view(-1, 1)
        else:
            y_cluster_tensor = torch.LongTensor(y_cluster)
        
        # Create dataset and dataloader
        train_dataset = TensorDataset(X_cluster_tensor, y_cluster_tensor)
        
        # Split train into train and validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        input_size = X_cluster.shape[2]  # Number of features
        model = LSTMModel(input_size, output_size).to(device)
        
        # Loss function and optimizer
        criterion = nn.BCELoss() if is_binary else nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y).item() * batch_X.size(0)
            
            val_loss /= len(val_dataset)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        models.append((cluster_id, model))
    
    # Evaluate each model on its respective test data
    results = {}
    for cluster_id, model in models:
        cluster_mask_test = clusters_test == cluster_id
        if np.sum(cluster_mask_test) < 1:  # Skip if no test samples
            continue
        
        X_cluster_test = X_test[cluster_mask_test]
        y_cluster_test = y_test[cluster_mask_test]
        
        # Convert to PyTorch tensors
        X_cluster_test_tensor = torch.FloatTensor(X_cluster_test).to(device)
        is_binary = len(set(y_train)) <= 2
        
        if is_binary:
            y_cluster_test_tensor = torch.FloatTensor(y_cluster_test).view(-1, 1).to(device)
        else:
            y_cluster_test_tensor = torch.LongTensor(y_cluster_test).to(device)
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            outputs = model(X_cluster_test_tensor)
            
            # Calculate loss
            criterion = nn.BCELoss() if is_binary else nn.CrossEntropyLoss()
            loss = criterion(outputs, y_cluster_test_tensor).item()
            
            # Calculate accuracy
            if is_binary:
                predictions = (outputs > 0.5).float()
                accuracy = (predictions == y_cluster_test_tensor).float().mean().item()
            else:
                _, predictions = torch.max(outputs, 1)
                accuracy = (predictions == y_cluster_test_tensor).float().mean().item()
        
        results[cluster_id] = {
            'loss': loss,
            'accuracy': accuracy,
            'sample_count': np.sum(cluster_mask_test)
        }
    
    return models, results

# Example usage
def main():
    df = load_file("PFC_con_4.csv")
    df = population_avg(df)
    df = clean_data(df)
    # Identify time columns (assuming they start from column 5)
    time_cols = df.columns[5:]
    
    # Extract the target column (assuming it's Group or similar in column 1)
    target_col = df.columns[1]
    
    # Perform DTW-based clustering on the 'during_tone' region
    clusters, feature_df, dtw_matrix = cluster_with_regional_dtw(
        df, time_cols, n_clusters=3, region='during_tone'
    )
    
    # Visualize the clusters
    # during_tone_fig = visualize_clusters(
    #     df, clusters, time_cols, 
    #     region_start_idx=10, region_end_idx=80  # during tone region
    # )
    # during_tone_fig.savefig('during_tone_clusters.png')
    
    # Visualize the dendrogram
    # visualize_dendrogram(dtw_matrix, feature_df['rat_id'])
    
    # Train RNN models using the clusters
    models, results = train_rnn_with_clusters(df, clusters, time_cols, target_col)
    
    # Print results
    print("Clustering Results:")
    for rat_id, cluster in clusters.items():
        print(f"Rat {rat_id}: Cluster {cluster}")
    
    print("\nModel Performance:")
    for cluster_id, metrics in results.items():
        print(f"Cluster {cluster_id} (n={metrics['sample_count']}): Accuracy = {metrics['accuracy']:.4f}, Loss = {metrics['loss']:.4f}")
    
    return clusters, feature_df, models, results

    
if __name__ == "__main__":
    main()