from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_segmentation_image_transform():
    """Image-only preprocessing (ImageNet normalization -- the U-Net encoder is ImageNet-pretrained).

    No resize: tiles are already saved at their target size by the fetch/tiling notebook.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class SegmentationTileDataset(Dataset):
    """Pairs of (RGB tile, class-index mask) PNGs, matched by filename across two directories.

    Masks must store raw class indices (0..num_classes-1) as pixel values, not a 0-255
    visualization -- use `pil_to_tensor` (not `ToTensor`, which would rescale indices to
    [0, 1]) and cast to long, exactly like the label tensors expected by CrossEntropyLoss.
    """

    def __init__(self, images_dir, masks_dir, image_transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_transform = image_transform or build_segmentation_image_transform()

        self.filenames = sorted(p.name for p in self.images_dir.glob("*.png"))
        missing = [name for name in self.filenames if not (self.masks_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} image tile(s) have no matching mask, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int):
        name = self.filenames[idx]

        image = Image.open(self.images_dir / name).convert("RGB")
        image = self.image_transform(image)

        mask = Image.open(self.masks_dir / name)
        mask = pil_to_tensor(mask).squeeze(0).long()

        return image, mask


def compute_class_pixel_weights(dataset: SegmentationTileDataset | Subset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights from the dataset's mask pixels, for a weighted loss.

    See docs/roadmap_segmentation.md §6: the #1 fix for the extreme class imbalance
    (parking/friche pixels were <0.3% of the total) that sank the first attempt. Pass the
    *train* split (e.g. the Subset returned by helpers.dataloaders.build_dataloaders), not
    the full dataset, so weights aren't influenced by val/test pixels.
    """
    if isinstance(dataset, Subset):
        base, filenames = dataset.dataset, [dataset.dataset.filenames[i] for i in dataset.indices]
    else:
        base, filenames = dataset, dataset.filenames

    counts = np.zeros(num_classes, dtype=np.int64)
    for name in filenames:
        mask = np.array(Image.open(base.masks_dir / name))
        class_ids, class_counts = np.unique(mask, return_counts=True)
        counts[class_ids] += class_counts

    counts = np.maximum(counts, 1)  # avoid division by zero for classes absent from this split
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)
