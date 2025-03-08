import torch
from torchvision import transforms
from model import ResNet50Base, ResNet50DFF
from utils import load_image, display_visualizations


def main():
    # Define transforms (should be the same as used in training)
    transform = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
    ])

    # Set up device and instantiate the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Let user specify an image path for evaluation
    image_path = "person.jpg"
    sample_img = load_image(image_path, transform).to(device)

    num_classes = 751  # <-- Adjust this value as needed
    model = ResNet50DFF(num_classes=num_classes).to(device)

    # Load the saved weights
    checkpoint = "dff_model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"Loaded model from {checkpoint}")

    # Run evaluation with and without DFF to extract features
    model.eval()
    with torch.no_grad():
        _, features_dff = model(sample_img)

    model = ResNet50Base(num_classes=num_classes).to(device)

    # Load the saved weights
    checkpoint = "base_model.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"Loaded model from {checkpoint}")

    # Run evaluation with and without DFF to extract features
    model.eval()
    with torch.no_grad():
        _, features_orig = model(sample_img)

    # Display all visualizations in one plot
    display_visualizations(features_orig, features_dff)


if __name__ == '__main__':
    main()
