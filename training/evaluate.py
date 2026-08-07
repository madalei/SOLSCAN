import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report


def evaluate_classifier(model, test_loader, device, classes, verbose: bool = True):
    """
        @param model: The neural network model to evaluate
        @param test_loader: DataLoader for the test set
        @param device: The device to which the model will be moved
        @param classes: List of class names
        @param verbose: Whether to print the classification report
        @return: Tuple of predicted labels and true labels (all_preds=[2, 7, 4, ...], all_labels=[4, 7, 7, ...])
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


def draw_confusion_matrix(all_labels, all_preds, classes, title=None):
    """
        Draw a confusion matrix using sklearn's ConfusionMatrixDisplay
        @param all_labels: list of true labels
        @param all_preds: list of predicted labels
        @param classes: list of class names
        @param title: optional title for the plot
    """
    _, ax = plt.subplots(figsize=(8, 8))

    ConfusionMatrixDisplay.from_predictions(
        all_labels,
        all_preds,
        display_labels=classes,
        xticks_rotation="vertical",
        ax=ax
    )

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()