import segmentation_models_pytorch as smp
from torch import nn


def build_unet(num_classes: int, device, encoder_name: str = "resnet34") -> nn.Module:
    """
    Build a U-Net for semantic segmentation, with an ImageNet-pretrained encoder.
    @param num_classes: Number of segmentation classes (including background).
    @param device: The device to which the model will be moved (e.g., 'cpu' or 'cuda').
    @param encoder_name: Backbone for the encoder (default resnet34, a good accuracy/speed tradeoff).
    @return: A U-Net model, moved to the specified device.
    """
    model = smp.Unet(
        encoder_name=encoder_name, # architecture de l'encodeur, defaut resnet34
        encoder_weights="imagenet", # poids pré-entraînés sur ImageNet pour l'encodeur, d'où viennent les poids initiaux de cette architecture ResNet34
        in_channels=3,
        classes=num_classes,
    )
    return model.to(device)
