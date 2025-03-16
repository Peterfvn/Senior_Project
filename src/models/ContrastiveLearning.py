import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np

"""SimCLR-style Contrastive Learning"""

# Feature Extractor (explore MLP/RNN/CNN/etc)
class FeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(FeatureExtractor, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2: # Change shape to [batch, sequence, # features (1 in this case)]
            x = x.unsqueeze(-1)

        _, h_n = self.rnn(x)
        h_n = h_n[-1]
        
        return self.fc(h_n)
    
# Projection head for contrastive loss
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim, hidden_dim=256, dropout=0.1):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

        self.fc3 = nn.Linear(hidden_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout1(x)

        residual = x
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        x += residual

        x = self.fc3(x)
        x = self.layer_norm(x)

        return x
    
class SimplerHead(nn.Module):
    def __init__(self, input_dim, projection_dim, hidden_dim=256):
        super(SimplerHead, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, projection_dim)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.layer_norm(x)
        return x
    

# Contrastive learning model
class ContrastiveModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, projection_dim):
        super(ContrastiveModel, self).__init__()
        self.encoder = FeatureExtractor(input_size, hidden_size, output_size)
        self.projection = SimplerHead(output_size, projection_dim)

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection(features)
        return projections, features
    
# Contrastive loss function
def contrastive_loss(z1, z2, device, temperature=0.7):
    """NT-Xent loss"""
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0) # Concat positive pairs

    # Similararity matrix
    sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

    # Labels (Positives on diagonal)
    labels = torch.arange(batch_size).to(device)
    labels = torch.cat([labels, labels], dim=0) # Duplicate for both augmented views

    # Apply contrastive loss
    loss = F.cross_entropy(sim_matrix / temperature, labels)
    return loss

def train_encoder(model, dataloader, optimizer, device, num_epochs=10, print_bool=True):
    import matplotlib.pyplot as plt # Temporary for visualization
    model.train()

    losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        for x1, x2 in dataloader: # x1 and x2 should be augmented views of same sample
            x1, x2 = x1.to(device), x2.to(device)

            # Embeddings
            z1, _ = model(x1)
            z2, _ = model(x2)

            # Contrastive loss
            loss = contrastive_loss(z1, z2, device)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if print_bool:
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

        losses.append(total_loss/len(dataloader))

    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Contrastive Loss Over Time")
    plt.tight_layout()

    plt.savefig("contrastive_loss.png")
    plt.close()

def augment_data(x):
    """Augmentation for contrastive learning"""
    def scaling(x, scale_factor=0.1):
        scale = 1.0 + (torch.randn(1, device=x.device) * scale_factor)
        return x * scale
    def jitter(x, sigma=0.05):
        return x + torch.randn_like(x) * sigma
    def batch_shuffle(x):
        return x[torch.randperm(x.shape[0])]
    x = jitter(x, sigma=0.05)
    x = scaling(x, scale_factor=0.1)

    return x

class ContrastiveDataset(Dataset):
    def __init__(self, data, indices, augment=True):
        self.data = data
        self.augment = augment
        self.indices = indices

        self.labels = torch.tensor(self.data.iloc[indices][self.data.columns[4]].values, dtype=torch.float32)

        self.features = torch.tensor(self.data.iloc[indices, 5:].values, dtype=torch.float32)

    def __len__(self):
        return len(self.indices)

    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        if self.augment: # Contrastive
            x1 = augment_data(x)
            x2 = augment_data(x)
            return x1, x2
        else:
            return x, y # Classification
    

class TestClassifier(nn.Module):
    def __init__(self, encoder, feature_dim, num_classes):
        super(TestClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features, _ = self.encoder(x)
        return self.classifier(features).squeeze()
    
def train_classifier(model, dataloader, optimizer, criterion, device, num_epochs=10):
    model.train()

    for epoch in range(num_epochs):
        correct = 0
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)

            # Squeeze the logits to match the shape of y, unsure why it is [batch_size, 1]
            logits = logits.squeeze()
            loss = criterion(logits, y)

            # Compute accuracy
            preds = (logits > 0).float()
            correct += (preds == y).sum().item()

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}, Accuracy: {correct/len(dataloader.dataset):.4f}")

def evaluate_classifier(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # Forawrd pass
            logits = model(x)
            preds = (logits > 0).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total
    return accuracy

def evaluate_encoder(model, dataloader, device):
    """Plotting embeddings using t-SNE"""
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import normalize

    # Use unaugmented test data
    model.eval()
    model = model.to(device)

    embeddings = []
    labels = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            embeddings.append(model.encoder(x).cpu().numpy())
            labels.append(y.cpu().numpy())

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings = np.concatenate(embeddings)
    embeddings = normalize(embeddings, axis=1)
    embeddings_2d = tsne.fit_transform(embeddings)
    labels = np.concatenate(labels)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title("t-SNE Visualization of Embeddings")
    plt.savefig("tsne_embeddings.png")
    plt.close()
