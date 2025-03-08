import torch
import torch.nn as nn
from torchvision.models import resnet50

from dff_module import DFF


class ResNet50DFF(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50DFF, self).__init__()
        self.base = resnet50(pretrained=True)
        self.dff1 = DFF(channels=256)  # After layer1, channels=256
        self.dff2 = DFF(channels=512)  # After layer2, channels=512

        # Remove final classifier and pooling
        del self.base.fc, self.base.avgpool

        # Add a classification head if training for classification
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = {}
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        features['layer1'] = x.detach().cpu()

        x = self.dff1(x)
        features['layer1_dff'] = x.detach().cpu()

        x = self.base.layer2(x)
        features['layer2'] = x.detach().cpu()

        x = self.dff2(x)
        features['layer2_dff'] = x.detach().cpu()

        # Global average pooling for classification
        x_pool = torch.mean(x, dim=[2, 3])
        logits = self.classifier(x_pool)
        return logits, features


class ResNet50Base(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Base, self).__init__()
        self.base = resnet50(pretrained=True)

        # Remove final classifier and pooling
        del self.base.fc, self.base.avgpool

        # Add a classification head if training for classification
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        features = {}
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        features['layer1'] = x.detach().cpu()

        x = self.base.layer2(x)
        features['layer2'] = x.detach().cpu()

        # Global average pooling for classification
        x_pool = torch.mean(x, dim=[2, 3])
        logits = self.classifier(x_pool)
        return logits, features