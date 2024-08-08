import torch
import torch.nn as nn
import torch.optim as optim
import timm  # Using timm for model creation and pretrained weights
from torch.utils.data import DataLoader, Dataset
import pickle
from tqdm import tqdm
import time

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if image.shape[0] == 1:
            # ResNet152 also expects 3 channel images
            image = image.repeat(3, 1, 1)
        return image, label

def load_data(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

train_dataset = load_data('train_dataset.pkl')
test_dataset = load_data('test_dataset.pkl')

train_data = CustomDataset(train_dataset)
test_data = CustomDataset(test_dataset)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

# Create ResNet-152 model, pretrained on ImageNet
model = timm.create_model('resnet50', pretrained=True, num_classes=len(load_data('class_labels.pkl')))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # You might need to tune the learning rate

def train_and_test(model, train_loader, test_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()

        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")
        # Implement test phase if necessary

train_and_test(model, train_loader, test_loader, criterion, optimizer)
torch.save(model.state_dict(), 'resnet50_tuned_model.pth')  # Saving the trained model
