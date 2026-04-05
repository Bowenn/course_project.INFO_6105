import torch
import torch.nn as nn
from torchvision import models


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    def __init__(self, num_blocks=4, base_filters=32, dropout=0.5, img_size=227):
        super().__init__()
        layers = []
        in_ch = 3
        out_ch = base_filters
        for _ in range(num_blocks):
            layers.append(ConvBlock(in_ch, out_ch))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 512)
        self.features = nn.Sequential(*layers)

        # compute flattened size after conv blocks
        feat_size = img_size
        for _ in range(num_blocks):
            feat_size = feat_size // 2
        flat_dim = in_ch * feat_size * feat_size

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def _make_pretrained(model_fn, weights, dropout):
    """Swap the final FC layer for binary classification."""
    model = model_fn(weights=weights)
    # freeze feature extractor
    for param in model.parameters():
        param.requires_grad = False
    # replace classifier head
    if hasattr(model, "fc"):  # resnet
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2),
        )
    elif hasattr(model, "classifier"):  # vgg
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2),
        )
    return model


def build_model(args):
    if args.model == "custom_cnn":
        return CustomCNN(
            num_blocks=args.num_conv_blocks,
            base_filters=args.base_filters,
            dropout=args.dropout,
            img_size=args.img_size,
        )

    pretrained_map = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        "vgg16":    (models.vgg16,    models.VGG16_Weights.DEFAULT),
    }
    model_fn, weights = pretrained_map[args.model]
    return _make_pretrained(
        model_fn,
        weights if args.pretrained else None,
        args.dropout,
    )
