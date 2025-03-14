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
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if x.dim() == 2: # Change shape to [batch, sequence, # features (1 in this case)]
            x = x.unsqueeze(-1)

        _, h_n = self.rnn(x)
        h_n = h_n[-1]
        
        return self.fc(h_n)
    
# Projection head for contrastive loss
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, projection_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, projection_dim)
        self.fc2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Contrastive learning model
class ContrastiveModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, projection_dim):
        super(ContrastiveModel, self).__init__()
        self.encoder = FeatureExtractor(input_size, hidden_size, output_size)
        self.projection = ProjectionHead(output_size, projection_dim)

    def forward(self, x):
        features = self.encoder(x)
        projections = self.projection(features)
        return projections, features
    
# Contrastive loss function
def contrastive_loss(z1, z2, device, temperature=0.5):
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

def train_encoder(model, dataloader, optimizer, device, num_epochs=10):
    model.train()

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

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

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
    print(f"Test Accuracy: {accuracy:.4f}")