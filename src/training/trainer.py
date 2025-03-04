import torch
from torch import optim

def train_model(model, train_data, criterion, optimizer, epochs, device):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_data:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")


"""Possible Implementation of Evaluate"""
# def evaluate(model, test_data, criterion, device):
#     model.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for inputs, targets in test_data:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             total_loss += loss.item()
#     return total_loss / len(test_data)


def evaluate(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy: {100 * correct / total}%')


"""Claude's idea about training using the clusters. Uses tensorflow, I'll look at this later"""
def train_rnn_with_clusters(df, clusters, time_cols, target_col, test_size=0.2):
    """Train an RNN model using cluster information."""
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.callbacks import EarlyStopping
    
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
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test, clusters_train, clusters_test = train_test_split(
        X, y, cluster_info, test_size=test_size, random_state=42, stratify=cluster_info
    )
    
    # Build RNN model with cluster-specific processing
    num_clusters = len(set(clusters.values()))
    models = []
    
    # Train separate models for each cluster or use cluster as a feature
    for cluster_id in range(num_clusters):
        # Get data for this cluster
        cluster_mask = clusters_train == cluster_id
        if np.sum(cluster_mask) < 2:  # Skip if too few samples
            continue
            
        X_cluster = X_train[cluster_mask]
        y_cluster = y_train[cluster_mask]
        
        # Define model
        model = Sequential([
            Input(shape=(X_cluster.shape[1], X_cluster.shape[2])),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid' if len(set(y_train)) <= 2 else 'softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy' if len(set(y_train)) <= 2 else 'sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train model
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(
            X_cluster, y_cluster,
            epochs=100,
            batch_size=8,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        models.append((cluster_id, model))
    
    # Evaluate each model on its respective test data
    results = {}
    for cluster_id, model in models:
        cluster_mask_test = clusters_test == cluster_id
        if np.sum(cluster_mask_test) < 1:  # Skip if no test samples
            continue
            
        X_cluster_test = X_test[cluster_mask_test]
        y_cluster_test = y_test[cluster_mask_test]
        
        loss, accuracy = model.evaluate(X_cluster_test, y_cluster_test, verbose=0)
        results[cluster_id] = {
            'loss': loss,
            'accuracy': accuracy,
            'sample_count': np.sum(cluster_mask_test)
        }
    
    return models, results