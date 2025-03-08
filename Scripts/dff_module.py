import torch
import torch.nn as nn

class DFF(nn.Module):
    def __init__(self, channels):
        super(DFF, self).__init__()
        # Reduce 2 channels (from avg and max pooling) to 1 using a 7x7 kernel
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # x: [B, C, H, W]
        x_fft = torch.fft.fft2(x)
        x_fft = torch.fft.fftshift(x_fft)

        # Compute pooling along channel dimension (magnitude only)
        avg_pool = torch.mean(x_fft.abs(), dim=1, keepdim=True)
        max_pool = torch.max(x_fft.abs(), dim=1, keepdim=True)[0]
        combined = torch.cat([avg_pool, max_pool], dim=1)
        mask = torch.sigmoid(self.conv(combined))

        # Apply mask to FFT result and perform inverse FFT
        x_filtered = x_fft * mask
        x_ifft = torch.fft.ifft2(torch.fft.ifftshift(x_filtered)).real
        return x_ifft