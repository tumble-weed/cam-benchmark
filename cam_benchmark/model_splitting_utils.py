import torch
import torchvision

# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# def _forward_impl(self, x: Tensor) -> Tensor:
#     # See note [TorchScript super()]
#     x = self.conv1(x)
#     x = self.bn1(x)
#     x = self.relu(x)
#     x = self.maxpool(x)

#     x = self.layer1(x)
#     x = self.layer2(x)
#     x = self.layer3(x)
#     x = self.layer4(x)

#     x = self.avgpool(x)
#     x = torch.flatten(x, 1)
#     x = self.fc(x)

#     return x
class SplitResnet50():
    def __init__(self, model,layername, pre_relu=False):
        self.model = model
        assert layername == 'layer4'
        self.layername = layername
        self.pre_relu = pre_relu
    def forward0(self,x):
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # out = self.model.layer4(x)

        # layer4: run all blocks except last fully
        *blocks, last = self.model.layer4
        for b in blocks:
            x = b(x)

        # run last bottleneck step by step
        identity = x
        if last.downsample is not None:
            identity = last.downsample(x)

        out = last.conv1(x)
        out = last.bn1(out)
        out = last.relu(out)

        out = last.conv2(out)
        out = last.bn2(out)
        out = last.relu(out)

        out = last.conv3(out)
        out = last.bn3(out)

        out = out + identity  # no relu here
        if not self.pre_relu:
            out = torch.nn.functional.relu(out) # apply the last relu

        return out

    def forward1(self, x):
        if self.pre_relu:
            x = torch.nn.functional.relu(x)
        x = self.model.avgpool(x)
        # for original resnet50, the fc is a Linear layer.
        # this might have been converted to a Conv2d layer in our modified model for torchray.
        if isinstance(self.model.fc,torch.nn.Linear):
            x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x
        
# https://github.com/pytorch/vision/blob/main/torchvision/models/vgg.py
class SplitVGG16:
    def __init__(self, model,layername, pre_relu=False):
        self.model = model
        assert layername == 'features.29'
        self.layername = layername
        self.pre_relu = pre_relu

    def forward0(self, x):
        # all feature layers except the last ReLU (and MaxPool)
        features = list(self.model.features)
        assert isinstance(features[-2],torch.nn.ReLU)
        assert isinstance(features[-1],torch.nn.MaxPool2d)
        if self.pre_relu:
            # stop before the ReLU at features[-2]
            for layer in features[:-2]:
                x = layer(x)
        else:
            for layer in features[:-1]:
                x = layer(x)
        return x

    def forward1(self, x):
        # apply remaining feature layers, then rest of the network
        if self.pre_relu:
            remaining = list(self.model.features)[-2:]  # ReLU + MaxPool
        else:
            remaining = list(self.model.features)[-1:]  # MaxPool only
        for layer in remaining:
            x = layer(x)
        # for original vgg16, the classifier is a Linear layer. 
        # this might have been converted to a Conv2d layer in our modified model for torchray.
        if isinstance(self.model.classifier[0],torch.nn.Linear):
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x
# https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py
# class SplitSwinT:
#     def __init__(self, model,layername):
#         self.model = model  # torchvision Swin-T
#         assert layername is None
#         self.layername = layername

#     def forward0(self, x):
#         # patch embedding
#         x = self.model.features[0](x)  # PatchEmbed

#         # Swin stages
#         for stage in self.model.features[1:-1]:
#             x = stage(x)

#         # last stage (BasicLayer): run all blocks, stop BEFORE final norm
#         last_stage = self.model.features[-1]
#         for blk in last_stage.blocks:
#             x = blk(x)

#         return x  # last feature map before norm / pooling (Grad-CAM target)

#     def forward1(self, x):
#         # apply final norm + pooling + head
#         x = self.model.norm(x)
#         x = self.model.avgpool(x)
#         x = torch.flatten(x, 1)
#         x = self.model.head(x)
#         return x
class SplitGoogLeNet():
    def __init__(self, model,layername):
        self.model = model
        assert layername == 'inception4d'
        self.layername = layername
        

    def forward0(self, x):
        # stem
        x = self.model.conv1(x)
        x = self.model.maxpool1(x)

        x = self.model.conv2(x)
        x = self.model.conv3(x)
        x = self.model.maxpool2(x)

        # inception 3
        x = self.model.inception3a(x)
        x = self.model.inception3b(x)
        x = self.model.maxpool3(x)

        # inception 4 up to 4d
        x = self.model.inception4a(x)
        x = self.model.inception4b(x)
        x = self.model.inception4c(x)
        x = self.model.inception4d(x)

        return x

    def forward1(self, x):
        # remaining inception blocks
        x = self.model.inception4e(x)
        x = self.model.maxpool4(x)

        x = self.model.inception5a(x)
        x = self.model.inception5b(x)

        # classifier head
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.dropout(x)
        x = self.model.fc(x)

        return x
def get_split_model(model,layername, pre_relu=False):
    if isinstance(model, torchvision.models.resnet.ResNet):
        return SplitResnet50(model,layername, pre_relu=pre_relu)
    elif isinstance(model, torchvision.models.vgg.VGG):
        return SplitVGG16(model,layername, pre_relu=pre_relu)
    elif isinstance(model, torchvision.models.GoogLeNet):
        return SplitGoogLeNet(model,layername)
    # elif isinstance(model, torchvision.models.swin_transformer.SwinTransformer):
    #     return SplitSwinT(model)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
def test_split_resnet50():
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()
    split_model = SplitResnet50(model,'layer4')
    x = torch.randn(1, 3, 224, 224)
    out0 = split_model.forward0(x)
    out1 = split_model.forward1(out0)
    print(out0.shape, out1.shape)
    # import ipdb;ipdb.set_trace()
    assert torch.allclose(model(x), out1)

def test_split_vgg16():
    model = torchvision.models.vgg16()
    model.eval()
    split_model = SplitVGG16(model,'features.29')
    x = torch.randn(1, 3, 224, 224)
    out0 = split_model.forward0(x)
    out1 = split_model.forward1(out0)
    print(out0.shape, out1.shape)
    assert torch.allclose(model(x), out1)

# def test_split_swin_transformer():
#     model = torchvision.models.swin_transformer.SwinTransformer()
#     model.eval()
#     split_model = SplitSwinT(model)
#     x = torch.randn(1, 3, 224, 224)
#     out0 = split_model.forward0(x)
#     out1 = split_model.forward1(out0)
#     print(out0.shape, out1.shape)
#     assert torch.allclose(model(x), out1)
def test_split_googlenet():
    model = torchvision.models.googlenet()
    model.eval()
    split_model = SplitGoogLeNet(model,'inception4d')
    x = torch.randn(1, 3, 224, 224)
    out0 = split_model.forward0(x)
    out1 = split_model.forward1(out0)
    print(out0.shape, out1.shape)
    assert torch.allclose(model(x), out1)
if __name__ == "__main__":
    test_split_resnet50()
    test_split_vgg16()
    # test_split_swin_transformer()
    test_split_googlenet()