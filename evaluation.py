import torch
import torch.nn as nn
from torchvision import models
import timm  # For accessing models like Vision Transformer and EfficientNet
from torch.utils.data import DataLoader, Dataset
import pickle
from sklearn.metrics import classification_report, accuracy_score

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        if image.shape[0] == 1:  # Convert grayscale to RGB if needed
            image = image.repeat(3, 1, 1)
        return image, label

def load_data(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

def evaluate_model(model, test_loader, model_name, class_names):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())

    accuracy = accuracy_score(y_true, y_pred) * 100  # Compute accuracy in percentage
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=False)
    print(f'Accuracy of {model_name} on the test images: {accuracy:.2f}%')
    return accuracy, report

# Load test dataset and class names
test_dataset = load_data('val_dataset.pkl')  # Adjust the path as necessary
class_names = load_data('class_names.pkl')  # Load class names
test_data = CustomDataset(test_dataset)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

def initialize_model(model_name, class_count, model_file):
    if 'vit' in model_name or 'efficientnet' in model_name:
        model = timm.create_model(model_name, pretrained=False, num_classes=class_count)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=False, aux_logits=False)  # Change aux_logits to match your saved model
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, class_count)
    else:
        model = getattr(models, model_name)(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, class_count)
    model.load_state_dict(torch.load(model_file))
    return model

# Models to evaluate
models_to_evaluate = {
    'resnet50': ('resnet50', 'resnet50_model.pth'),
    'resnet101': ('resnet101', 'resnet101_model.pth'),
    'resnet152': ('resnet152', 'resnet152_model.pth'),
    'googlenet': ('googlenet', 'googlenet_model.pth'),
    'efficientnet_b0': ('efficientnet_b0', 'efficientnet_b0_model.pth')
}

num_classes = len(class_names)  # Assume class names array's length represents number of classes

with open('evaluation_results.txt', 'w') as f:
    for model_key, (model_name, model_file) in models_to_evaluate.items():
        print(f"Evaluating {model_name}")
        model = initialize_model(model_name, num_classes, model_file)
        accuracy, metrics = evaluate_model(model, test_loader, model_name, class_names)
        f.write(f"{model_name} Model Evaluation:\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write("Classification Report:\n")
        f.write(metrics + "\n")
        f.write("\n")
