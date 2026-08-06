from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_efficientnet_b0_classifier(num_classes: int, device) -> nn.Module:
    """
    Build an EfficientNet-B0 model for image classification.
    @param num_classes: Number of output classes for the classifier.
    @param device: The device to which the model will be moved (e.g., 'cpu' or 'cuda').
    @return: An EfficientNet-B0 model with the final classifier layer modified to match the number of classes, moved to the specified device.
    """
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    # le classifier d'EfficientNet est un Sequential(Dropout, Linear) -> on remplace la couche Linear (index 1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(device)


# -----------------------------------------
# functions for freez / unfreeze strategies
#
def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
