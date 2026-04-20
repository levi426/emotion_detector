def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    epochs = params["train"]["epochs"]
    learning_rate = params["train"]["learning_rate"]
    batch_size = params["data"]["batch_size"]

    mlflow.set_experiment("emotion-classification")

    train_loader, test_loader, classes = load_data(batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    for name, param in model.named_parameters():
        if "layer3" not in name and "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_loss = float("inf")

    with mlflow.start_run():
        mlflow.log_params(params)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            running_loss /= len(train_loader)
            print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")
            mlflow.log_metric("train_loss", running_loss, step=epoch)

            scheduler.step()

            if running_loss < best_loss:
                best_loss = running_loss
                os.makedirs("models", exist_ok=True)
                torch.save(model.state_dict(), "models/fer_model.pth")

        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()