import matplotlib.pyplot as plt
import numpy as np


def visualize_features(feature_map, title):
    plt.figure(figsize=(10, 5))
    plt.imshow(feature_map[0, 0].cpu().numpy(), cmap='viridis')
    plt.title(title)
    plt.axis('off')
    plt.show()


def plot_fft(feature_map, title):
    fft = np.fft.fft2(feature_map)
    fft_shift = np.fft.fftshift(fft)
    magnitude = 20 * np.log(np.abs(fft_shift) + 1)
    plt.imshow(magnitude, cmap='viridis')
    plt.title(title)
    plt.axis('off')
