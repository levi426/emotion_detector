def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    batch_size = params["evaluate"]["batch_size"]

    _, test_loader, classes = load_data(batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    model.load_state_dict(torch.load("models/fer_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds) * 100

    os.makedirs("eval", exist_ok=True)
    with open("eval/metrics.json", "w") as f:
        json.dump({"accuracy": accuracy}, f, indent=4)

    print(f"Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    main()