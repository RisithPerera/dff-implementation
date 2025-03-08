import matplotlib.pyplot as plt
import numpy as np


def display_visualizations(features_orig, features_dff):
    # Extract the feature maps (assumes [batch, channels, height, width])
    orig_layer1 = features_orig['layer1'][0, 0].cpu().numpy()
    dff_layer1 = features_dff.get('layer1_dff', features_orig['layer1'])[0, 0].cpu().numpy()
    orig_layer2 = features_orig['layer2'][0, 0].cpu().numpy()
    dff_layer2 = features_dff.get('layer2_dff', features_orig['layer2'])[0, 0].cpu().numpy()

    # Function to compute FFT magnitude
    def compute_fft(img):
        fft = np.fft.fft2(img)
        fft_shift = np.fft.fftshift(fft)
        magnitude = 20 * np.log(np.abs(fft_shift) + 1)
        return magnitude

    fft_orig = compute_fft(orig_layer1)
    fft_dff = compute_fft(dff_layer1)

    # Create a figure with 2 rows:
    # First row: 4 subplots for feature maps.
    # Second row: 2 subplots for FFT images.
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # First row: Feature maps
    axes[0, 0].imshow(orig_layer1, cmap='viridis')
    axes[0, 0].set_title("Layer1 Original")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(dff_layer1, cmap='viridis')
    axes[0, 1].set_title("Layer1 DFF")
    axes[0, 1].axis('off')

    axes[0, 2].imshow(orig_layer2, cmap='viridis')
    axes[0, 2].set_title("Layer2 Original")
    axes[0, 2].axis('off')

    axes[0, 3].imshow(dff_layer2, cmap='viridis')
    axes[0, 3].set_title("Layer2 DFF")
    axes[0, 3].axis('off')

    # Second row: FFT images for layer1
    axes[1, 0].imshow(fft_orig, cmap='viridis')
    axes[1, 0].set_title("Layer1 FFT Original")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(fft_dff, cmap='viridis')
    axes[1, 1].set_title("Layer1 FFT DFF")
    axes[1, 1].axis('off')

    # Hide unused subplots in the second row
    axes[1, 2].axis('off')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.show()


def load_image(image_path, transform):
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    # Add batch dimension: [1, C, H, W]
    return image.unsqueeze(0)