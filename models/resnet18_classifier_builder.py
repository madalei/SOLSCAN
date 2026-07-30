from torch import  nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18_classifier(num_classes: int, device) -> nn.Module:
    """
    Build a ResNet-18 model for image classification.
    @param num_classes: Number of output classes for the classifier.
    @param device: The device to which the model will be moved (e.g., 'cpu' or 'cuda').
    @return: A ResNet-18 model with the final fully connected layer modified to match the number of classes, moved to the specified device.
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # remplace la tête fully-connected pour matcher le nombre de classes du dataset
    # Replace the final fully connected layer to match the number of classes in EuroSAT
    # fc layer has in_features=512 for ResNet18, we need to set out_features=len(classes) (numbers of classes in EuroSAT is 10)
    # fc layer is the last layer of the model, we can access it via model.fc (final classifier layer)
    # "la dernière couche doit produire un vecteur de <num_classes> nombres au lieu de 1000" 
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)


# -----------------------------------------
# functions for freez / unfreeze strategies
#
def freeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


