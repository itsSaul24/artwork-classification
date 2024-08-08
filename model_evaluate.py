import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
import pickle
from tqdm import tqdm

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if image.shape[0] == 1:  # Convert grayscale to RGB
            image = image.repeat(3, 1, 1)
        return image, label

def load_data(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

def initialize_model(num_classes, pretrained=True):
    model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)
    return model

def evaluate_model(model, test_loader, criterion):
    model.eval()
    total = 0
    correct = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            running_loss += loss.item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            images, labels = data
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def grid_search_hyperparameters(grid_params, train_data, test_data, num_classes, result_filename='grid_search_results.txt'):
    with open(result_filename, 'w') as file:
        file.write("Batch Size, Learning Rate, Test Loss, Test Accuracy\n")
        for lr in grid_params['learning_rates']:
            for batch_size in grid_params['batch_sizes']:
                train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
                test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

                model = initialize_model(num_classes)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)

                print(f"Training with batch size {batch_size} and learning rate {lr}")
                train_model(model, train_loader, criterion, optimizer)

                loss, accuracy = evaluate_model(model, test_loader, criterion)
                result = f"{batch_size}, {lr}, {loss:.4f}, {accuracy:.2f}%\n"
                file.write(result)
                print(f"Results for BS={batch_size}, LR={lr}: Loss={loss:.4f}, Accuracy={accuracy:.2f}%")

# Example usage:
num_classes = len(load_data('class_labels.pkl'))
train_dataset = CustomDataset(load_data('train_dataset.pkl'))
test_dataset = CustomDataset(load_data('test_dataset.pkl'))

grid_params = {
    'learning_rates': [0.001, 0.0001],
    'batch_sizes': [16, 32]
}

grid_search_hyperparameters(grid_params, train_dataset, test_dataset, num_classes)
