from dataclasses import dataclass
from typing import Callable
from models.resnet18_classifier_builder import build_resnet18_classifier
from models.unet_builder import build_unet

# TODO for later benchmark experiment, to easy change param config and test them

@dataclass
class PipelineConfig:
    """
    Configuration for a machine learning experiment.
    @param name: Name of the experiment.
    @param build_model: Callable function to build the model.
    @param num_classes: Number of output classes for the model.
    @param lr: Learning rate for the optimizer (default: 1e-4).
    @param epochs: Number of epochs to train the model (default: 5).
    
    usage example:
    in this config file, we can define a config for a ResNet18 model for EuroSAT dataset:
    RESNET18_BASELINE = ExperimentConfig(
        name="resnet18_baseline",       
        build_model=build_resnet18_classifier,
        num_classes=10,
        lr=1e-4,
        epochs=5,
    )

    in another place, we can use this config to build the model and train it:
    config = RESNET18_BASELINE
    model = config.build_model(num_classes=config.num_classes, device=device)   
    """
    name: str
    build_model: Callable
    num_classes: int
    lr: float = 1e-4
    epochs: int = 5


RESNET18_BASELINE = PipelineConfig(
    name="resnet18_baseline",
    build_model=build_resnet18_classifier,
    num_classes=10,
)

# 5 classes: fond, parking, industriel/commercial, friche, residentiel -- see docs/roadmap_segmentation.md §1
UNET_CONFIG = PipelineConfig(
    name="unet_landuse",
    build_model=build_unet,
    num_classes=5,
    lr=1e-4,
    epochs=20,
)