import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import Market1501Dataset
from model import ResNet50Base, ResNet50DFF


def train_model(model, train_loader, device, num_epochs=2):
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    total_batches = len(train_loader)
    model.train()

    for epoch in range(num_epochs):
        for batch_idx, (images, pids) in enumerate(train_loader):
            images, pids = images.to(device), pids.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, pids)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                percent_complete = 100 * batch_idx / total_batches
                print(f'Epoch: {epoch + 1}/{num_epochs} | {percent_complete:.2f}% complete | Loss: {loss.item():.4f}')

    return model


def main():
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
    ])

    # Download Link: https://www.kaggle.com/api/v1/datasets/download/pengcw1/market-1501
    dataset_root = '../DataSets/Market-1501-v15.09.15/bounding_box_train'
    dataset = Market1501Dataset(root=dataset_root, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # Set up device and instantiate the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(dataset.pid2label)

    model = ResNet50DFF(num_classes=num_classes).to(device)

    # Train the model without DFF
    trained_model = train_model(model, train_loader, device, num_epochs=2)

    # Save the model weights
    torch.save(trained_model.state_dict(), "dff_model.pth")
    print("Model saved to dff_model.pth")

    model = ResNet50Base(num_classes=num_classes).to(device)

    # Train the model without DFF
    trained_model = train_model(model, train_loader, device, num_epochs=2)

    # Save the model weights
    torch.save(trained_model.state_dict(), "base_model.pth")
    print("Model saved to base_model.pth")


if __name__ == '__main__':
    main()
