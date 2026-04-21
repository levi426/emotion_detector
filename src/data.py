import os
import yaml
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.utils import split_train_test

def load_data(batch_size=None, test_ratio=None):
    if batch_size is None or test_ratio is None:
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        if batch_size is None:
            batch_size = params["data"]["batch_size"]
        if test_ratio is None:
            test_ratio = params["data"]["test_ratio"]
    os.makedirs("data_split/train", exist_ok=True)
    os.makedirs("data_split/test", exist_ok=True)
    if not os.listdir("data_split/train") or not os.listdir("data_split/test"):
        split_train_test("data", "data_split", test_ratio=test_ratio)

    train_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder("data_split/train", transform=train_transform)
    test_dataset = datasets.ImageFolder("data_split/test", transform=test_transform)

    print("Train samples:", len(train_dataset))
    print("Test samples:", len(test_dataset))
    print("Classes:", train_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.classes

if __name__ == "__main__":
    load_data()
