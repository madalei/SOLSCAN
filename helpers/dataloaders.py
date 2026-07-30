import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import EuroSAT

DATA_DIR = Path.cwd().parent / "data"

def build_eurosat_transform(tile_size: int = 64):
    """Preprocessing pipeline (resize/tensor/ImageNet normalization)"""
    return transforms.Compose([
        transforms.Resize(tile_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_eurosat_data():
    """Fetch EuroSAT dataset"""

    # 1/ Define the data transforms (preprocess to apply to the images when loading the dataset)
    # ImageNet normalization since we fine-tune a pretrained ResNet18
    transform = build_eurosat_transform()

    # 2/ Load the EuroSAT dataset with the defined transforms
    # dataset attrs: [classes, ...?] accesible like dataset.classes
    dataset = EuroSAT(root=str(DATA_DIR), download=True, transform=transform)

    return dataset

def build_dataloaders(dataset, batch_size: int, train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 42):
    """
        Split a dataset into train/validation/test dataloader subsets
        @return a tuple of 3 DataLoaders: train, validation and test
    """

    # Define the random number generator wich seed is fixed to ensure reproducibility (rendre le tirage aléatoire reproductible)
    generator = torch.Generator().manual_seed(seed)

    # 4/ Split the dataset into training, validation and test sets
    train_size = int(train_frac * len(dataset))
    val_size = int(val_frac * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# ============================================================================
# ROLES DES 3 SPLITS : train / val / test
# ============================================================================
#
# TRAIN (train_loader)
#   -> Le seul set sur lequel les poids du modele sont reellement modifies
#      (loss.backward() + optimizer.step()). Le modele "apprend" dessus,
#      epoch apres epoch.
#
# VAL (val_loader)
#      Son rôle : mesurer, après chaque epoch, si le modèle généralise bien à des données qu'il n'a pas apprises par cœur.
#   -> Jamais utilise pour entrainer (pas de backward/step), seulement pour
#      MESURER la performance apres chaque epoch.
#   -> Sert a detecter l'overfitting : si train_loss baisse mais val_loss
#      remonte, le modele memorise le train set au lieu de generaliser.
#   -> Sert aussi a PRENDRE DES DECISIONS pendant le dev : combien d'epochs,
#      quel lr, quelle architecture (ResNet vs U-Net vs DeepLab...), quel
#      checkpoint garder.
#
# TEST (test_loader)
#   -> Jamais touche avant la toute fin, une seule fois, une fois tous les
#      choix (hyperparametres, architecture, epochs) deja figes.
#   -> Donne une mesure HONNETE de la performance finale.
#
# POURQUOI 3 SPLITS ET PAS JUSTE TRAIN/TEST ?
#   -> Si on utilise le meme set pour ajuster les choix ET mesurer la
#      performance finale, on finit par choisir les hyperparametres qui
#      collent le mieux a CE set precis -> la mesure finale devient
#      optimiste et ne reflete plus la vraie capacite de generalisation.
#   -> Le test set reste "propre" (jamais utilise pour decider quoi que ce
#      soit) pour donner une estimation fiable de comment le modele se
#      comportera sur des donnees reelles jamais vues (ex: nouvelles images
#      satellite a analyser).
# ============================================================================
