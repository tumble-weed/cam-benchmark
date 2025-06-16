import torch
import torch.nn as nn

from torchvision.models import ResNet, VGG, GoogLeNet
from timm.models.swin_transformer import SwinTransformer


# -------- Truncated Definitions (Start AFTER specific layer) --------

class ResNet50AfterLayer4(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        # start from avgpool → fc
        self.avgpool = original_model.avgpool
        self.fc = original_model.fc

    def forward(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class VGG16AfterFeatures29(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.rest_features = nn.Sequential(*list(original_model.features.children())[30:])
        self.avgpool = original_model.avgpool
        self.classifier = original_model.classifier

    def forward(self, x):
        x = self.rest_features(x)  
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



class GoogleNetAfterInception4d(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = nn.Sequential(
            original_model.inception4e,
            original_model.maxpool4,
            original_model.inception5a,
            original_model.inception5b,
            original_model.avgpool,
            nn.Flatten(),
            original_model.dropout,
            original_model.fc,
        )

    def forward(self, x):
        return self.model(x)


class SwinTAfterLayers3Blocks1Norm2(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.layers = nn.Sequential(
            *original_model.layers[3].blocks[2:],  # remaining blocks in last stage
            original_model.norm,
        )
        self.head = original_model.head

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        x = x.mean(dim=1)  # global average pool (B, C)
        x = self.head(x)
        return x

# -------- Dispatcher Function --------

def get_truncated_model(model, layer_name):
    """
    Returns the part of the model that comes AFTER the specified layer.
    Uses isinstance for robustness.
    """
    model.eval()

    if isinstance(model, ResNet):
        assert layer_name == 'layer4', f"Unsupported layer {layer_name} for ResNet"
        return ResNet50AfterLayer4(model)

    elif isinstance(model, VGG):
        assert layer_name == 'features_29', f"Unsupported layer {layer_name} for VGG"
        return VGG16AfterFeatures29(model)

    elif isinstance(model, GoogLeNet):
        assert layer_name == 'inception4d', f"Unsupported layer {layer_name} for GoogLeNet"
        return GoogleNetAfterInception4d(model)

    elif isinstance(model, SwinTransformer):
        assert layer_name == 'layers_3_blocks_1_norm', f"Unsupported layer {layer_name} for SwinTransformer"
        return SwinTAfterLayers3Blocks1Norm2(model)

    else:
        raise NotImplementedError(
            f"Truncation for model of type {type(model)} after layer '{layer_name}' not implemented."
        )
