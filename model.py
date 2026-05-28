import torch.nn as nn
from torchvision import models


class InsulatorClassifier(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=2):
        super().__init__()
        self.model_name = model_name

        if model_name == 'resnet18':
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            in_features = backbone.fc.in_features
            backbone.fc = nn.Linear(in_features, num_classes)
            self.backbone = backbone

        elif model_name == 'efficientnet_b0':
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            in_features = backbone.classifier[1].in_features
            backbone.classifier[1] = nn.Linear(in_features, num_classes)
            self.backbone = backbone

        elif model_name == 'mobilenet_v2':
            backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            in_features = backbone.classifier[1].in_features
            backbone.classifier[1] = nn.Linear(in_features, num_classes)
            self.backbone = backbone

        else:
            raise ValueError(
                f'Неподдерживаемая модель: {model_name}. '
                f'Доступно: resnet18, efficientnet_b0, mobilenet_v2'
            )

    def forward(self, x):
        return self.backbone(x)
