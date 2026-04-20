import os
import yaml
import json
import torch
import torch.nn as nn     
import numpy as np
from torchvision import models
from torchvision.models import ResNet18_Weights
from src.data import load_data
from sklearn.metrics import classification_report, accuracy_score

with open("params.yaml") as f:
    params = yaml.safe_load(f)

batch_size = params["evaluate"]["batch_size"]

_, test_loader = load_data(batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 7)
model.load_state_dict(torch.load("models/fer_model.pth", map_location=device))
model = model.to(device)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
print(f"Test Accuracy: {accuracy:.2f}%")

report = classification_report(all_labels, all_preds, output_dict=True)
print(classification_report(all_labels, all_preds, digits=4))

os.makedirs("eval", exist_ok=True)
with open("eval/metrics.json", "w") as f:
    json.dump({"accuracy": accuracy, "report": report}, f, indent=4)

print("Saved metrics to eval/metrics.json")
