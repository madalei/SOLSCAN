import copy

import torch
from segmentation_models_pytorch.losses import DiceLoss
from torch import nn


class SegEngine:
    """Training engine for the U-Net landuse segmentation model.

    Loss is Dice + CrossEntropy (optionally class-weighted) -- fixed rather than swappable
    like training.engine.Engine's loss_fn, since this combo is what docs/roadmap_segmentation.md
    §4 validated. Per-class IoU is tracked every epoch, not just the mean: on the first
    attempt the mean IoU looked reasonable while the two rare classes (parking, friche)
    were essentially at 0 -- see docs/roadmap_segmentation.md §5.
    """

    def __init__(self, device, optimizer, num_classes: int, class_weights: torch.Tensor | None = None):
        self.device = device
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.dice_loss = DiceLoss(mode="multiclass", from_logits=True)
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)

        # Populated by train_model -- the val-loss-minimizing epoch's weights, since with a
        # dataset this small the model keeps overfitting well past that point (train loss
        # keeps falling while val loss climbs back up), so the *last* epoch is usually not
        # the one worth deploying. See docs/roadmap_segmentation.md and the val loss curve
        # in notebooks/train_unet_landuse.ipynb section 5/6 for the observed pattern.
        self.best_val_loss = float("inf")
        self.best_epoch: int | None = None
        self.best_state_dict: dict | None = None

    def display_info(self):
        print(f"device: {self.device}, num_classes: {self.num_classes}, ce weight: {self.ce_loss.weight}, optimizer: {self.optimizer}")

    def _loss(self, logits, masks):
        return self.dice_loss(logits, masks) + self.ce_loss(logits, masks)

    def _confusion_counts(self, logits, masks):
        """Per-class intersection/union pixel counts for this batch (int64)."""
        preds = logits.argmax(1)
        intersection = torch.zeros(self.num_classes, dtype=torch.int64, device=self.device)
        union = torch.zeros(self.num_classes, dtype=torch.int64, device=self.device)
        for c in range(self.num_classes):
            pred_c = preds == c
            mask_c = masks == c
            intersection[c] = (pred_c & mask_c).sum()
            union[c] = (pred_c | mask_c).sum()
        return intersection, union

    def train_epoch(self, model, dataloader) -> float:
        model.train()
        total_loss, total = 0.0, 0
        for images, masks in dataloader:
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            logits = model(images)
            loss = self._loss(logits, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            total += images.size(0)
        return total_loss / total

    def eval_epoch(self, model, dataloader) -> tuple[float, torch.Tensor]:
        """@return (avg_loss, iou_per_class) -- iou_per_class is NaN for classes absent from this split."""
        model.eval()
        total_loss, total = 0.0, 0
        intersection = torch.zeros(self.num_classes, dtype=torch.int64, device=self.device)
        union = torch.zeros(self.num_classes, dtype=torch.int64, device=self.device)

        with torch.no_grad():
            for images, masks in dataloader:
                images, masks = images.to(self.device), masks.to(self.device)
                logits = model(images)
                loss = self._loss(logits, masks)

                total_loss += loss.item() * images.size(0)
                total += images.size(0)

                batch_inter, batch_union = self._confusion_counts(logits, masks)
                intersection += batch_inter
                union += batch_union

        union = union.float()
        iou_per_class = torch.where(union > 0, intersection.float() / union, torch.full_like(union, float("nan")))
        return total_loss / total, iou_per_class.cpu()

    # Note: Pas d'accuracy actuellement dans SegEngine, 
    # vu le déséquilibre extrême déjà documenté (fond = large majorité des pixels), 
    # l'accuracy pixel serait trompeuse ici — un modèle qui prédit "fond partout" obtiendrait ~99% d'accuracy 
    # tout en étant inutile (0% sur parking/friche)
    def train_model(self, model, train_loader, val_loader, epochs: int, class_names: list[str] | None = None) -> dict:
        history = {"train_loss": [], "val_loss": [], "val_iou_per_class": [], "val_mean_iou": []}
        names = class_names or [f"class_{i}" for i in range(self.num_classes)]

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(model, train_loader)
            val_loss, val_iou = self.eval_epoch(model, val_loader)
            mean_iou = float(torch.nanmean(val_iou))

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_state_dict = copy.deepcopy(model.state_dict())

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_iou_per_class"].append(val_iou.tolist())
            history["val_mean_iou"].append(mean_iou)

            iou_str = ", ".join(f"{n}={v:.3f}" for n, v in zip(names, val_iou.tolist()))
            print(
                f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - "
                f"Mean IoU: {mean_iou:.4f} ({iou_str})"
            )

        print(f"Best epoch: {self.best_epoch}/{epochs} (val loss {self.best_val_loss:.4f}) -- see self.best_state_dict")
        return history
