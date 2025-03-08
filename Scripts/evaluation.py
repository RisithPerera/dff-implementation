import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import Market1501Dataset
from model import ResNet50
from utils import visualize_features, plot_fft

def main():
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
    ])

    # Set dataset root (adjust as necessary)
    # Download Link: https://www.kaggle.com/api/v1/datasets/download/pengcw1/market-1501
    dataset_root = '../DataSets/Market-1501-v15.09.15/bounding_box_train'
    dataset = Market1501Dataset(root=dataset_root, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    # Set up device and instantiate the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(dataset.pid2label)
    model = ResNet50(num_classes=num_classes).to(device)

    # Load the saved weights
    model.load_state_dict(torch.load("model_final.pth"))
    print("Loaded model from model_final.pth")

    # Evaluate and visualize features
    model.eval()
    with torch.no_grad():
        sample_img, _ = next(iter(train_loader))
        sample_img = sample_img[0:1].to(device)

        # Get features with DFF enabled
        _, features_dff = model(sample_img, use_dff=True)
        # Get features with DFF disabled (original features)
        _, features_orig = model(sample_img, use_dff=False)

    # Visualize layer1 features
    visualize_features(features_orig['layer1'], "Layer1 Features (Original)")
    visualize_features(features_dff.get('layer1_dff', features_orig['layer1']), "Layer1 Features (DFF Processed)")

    # Visualize layer2 features
    visualize_features(features_orig['layer2'], "Layer2 Features (Original)")
    visualize_features(features_dff.get('layer2_dff', features_orig['layer2']), "Layer2 Features (DFF Processed)")

    # Additional visualization: FFT magnitude of feature maps
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plot_fft(features_orig['layer1'][0, 0].numpy(), "Original Layer1 FFT")
    plt.subplot(1, 2, 2)
    plot_fft(features_dff.get('layer1_dff', features_orig['layer1'])[0, 0].numpy(), "DFF Layer1 FFT")
    plt.show()


if __name__ == '__main__':
    main()
