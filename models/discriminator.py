import torch
import torch.nn as nn
from utils import LearnableSigmoid
from dimensions import *

class Discriminator(nn.Module):
    def __init__(self, depth_feature_maps, in_channel=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv2d(in_channel, depth_feature_maps, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(depth_feature_maps, affine=True),
            nn.PReLU(depth_feature_maps),
            nn.utils.spectral_norm(
                nn.Conv2d(depth_feature_maps, depth_feature_maps * 2, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(depth_feature_maps * 2, affine=True),
            nn.PReLU(2 * depth_feature_maps),
            nn.utils.spectral_norm(
                nn.Conv2d(depth_feature_maps * 2, depth_feature_maps * 4, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(depth_feature_maps * 4, affine=True),
            nn.PReLU(4 * depth_feature_maps),
            nn.utils.spectral_norm(
                nn.Conv2d(depth_feature_maps * 4, depth_feature_maps * 8, (4, 4), (2, 2), (1, 1), bias=False)
            ),
            nn.InstanceNorm2d(depth_feature_maps * 8, affine=True),
            nn.PReLU(8 * depth_feature_maps),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(depth_feature_maps * 8, depth_feature_maps * 4)),
            nn.Dropout(DISCRIMINATOR_DROPOUT_RATE),
            nn.PReLU(4 * depth_feature_maps),
            nn.utils.spectral_norm(nn.Linear(depth_feature_maps * 4, 1)),
            LearnableSigmoid(1),
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.layers(xy)
