import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def evaluate_classifier(model, test_loader, device, classes, verbose: bool = True):
    """
        output:
        all_preds  = [4, 4, 7, 2, 9, 4, 0, ...]   # classes predites
        all_labels = [4, 7, 7, 2, 9, 4, 0, ...]   # vraies classes 

    """
    model.eval()                                    # mode évaluation
    all_preds, all_labels = [], []
    with torch.no_grad():                          # pas de calcul de gradients
        for images, labels in test_loader:         # Boucle sur test_loader
            images = images.to(device)

            # model(images) -> (batch_size, 10) logits bruts par image
            # .argmax(1)    -> index du logit le plus eleve par ligne = classe predite (0-9)
            # .cpu()        -> rapatrie du device (mps/cuda) vers CPU, requis par sklearn
            preds = model(images).argmax(1).cpu()

            # accumule les prédictions/labels de tous les batchs dans deux listes Python simples 
            # (pas des tenseurs), au fur et à mesure qu'on avance
            all_preds.extend(preds.tolist())     
            all_labels.extend(labels.tolist())

    if verbose: print(classification_report(all_labels, all_preds, target_names=classes)) 

    return all_preds, all_labels


def draw_confusion_matrix(all_labels, all_preds, classes):
    _, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(all_labels, all_preds, display_labels=classes, xticks_rotation="vertical", ax=ax)
    plt.tight_layout()
    plt.show()
