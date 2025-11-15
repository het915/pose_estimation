"""Image encoder for FoundationPose."""

import torch
import torch.nn as nn
import torchvision.models as models


class ImageEncoder(nn.Module):
    """CNN encoder for extracting image features."""

    def __init__(
        self,
        backbone: str = "resnet50",
        feature_dim: int = 256,
        pretrained: bool = True,
    ):
        """
        Args:
            backbone: Backbone architecture
            feature_dim: Output feature dimension
            pretrained: Whether to use pretrained weights
        """
        super(ImageEncoder, self).__init__()

        self.backbone_name = backbone
        self.feature_dim = feature_dim

        # Load backbone
        if backbone == "resnet50":
            backbone_model = models.resnet50(pretrained=pretrained)
            # Remove final FC layer
            self.backbone = nn.Sequential(*list(backbone_model.children())[:-2])
            backbone_channels = 2048
        elif backbone == "resnet34":
            backbone_model = models.resnet34(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(backbone_model.children())[:-2])
            backbone_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Feature projection
        self.projection = nn.Sequential(
            nn.Conv2d(backbone_channels, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: (B, C_in, H, W) input image

        Returns:
            features: (B, feature_dim, H', W') features
        """
        # Extract features
        features = self.backbone(x)  # (B, C, H', W')

        # Project to desired dimension
        features = self.projection(features)  # (B, feature_dim, H', W')

        return features
