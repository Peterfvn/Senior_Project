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