import torch

class Engine:
    """
    Represents the training engine with a device, loss_fn (aka criterion), and optimizer.
    @param device: The device to run the training on (e.g., 'cpu' or 'cuda').
    @param loss_fn: The loss function used for training (instance of torch.nn.Module). A.K.A criterion
    @param optimizer: The optimizer used for updating model weights (instance of torch.optim.Optimizer).
    """
    def __init__(self, device, loss_fn, optimizer):
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer  

    def display_info(self):
        print(f"device: {self.device}, loss_fn: {self.loss_fn}, optimizer: {self.optimizer}")
  

    # Define the training and evaluation loop
    #
    # We could split the training and evaluation loop into two separate functions, 
    # "DRY" version here: combine them into a single function that takes a boolean argument to indicate whether we are training or evaluating. 
    # because the only difference between training and evaluation is whether we compute gradients and update weights (train) or not (eval).

    def run_epoch(self, model, dataloader, train: bool):
        # if true, set model to training mode, else set to evaluation mode
        # if false, autograd is disabled, so no gradients are computed and no weights are updated
        # if false, it s same as model.eval() and torch.no_grad() (!! magic code !!)

        """
        Run a single epoch of training or evaluation.
        @param model: The neural network model
        @param dataloader: DataLoader object for the dataset (train or validation)
        @param train: Boolean indicating whether to train or evaluate
        @return: Tuple of average loss and accuracy for the epoch   

        note: C'est l'entraînement (la loss) qui crée la correspondance entre les images et les labels
        """
        
        model.train(train) 
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train):
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)

                if train:
                    self.optimizer.zero_grad()

                outputs = model(images)                 # vecteur de 10 nombres, sans signification au départ
                loss = self.loss_fn(outputs, labels)    # compare à l'entier "vrai label" (ex: 4 pour Industrial)

                if train:
                    loss.backward()                     # calcule comment ajuster les poids
                    self.optimizer.step()               # ajuste les poids

                total_loss += loss.item() * images.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += images.size(0)

        return total_loss / total, correct / total

    # Train the model for a number of epochs, and evaluate on the validation set after each epoch
    def train_model(self, model, train_loader, val_loader, epochs):
        """
        Train the model for a specified number of epochs, and evaluate on the validation set after each epoch.

        @param model        : The neural network model to train
        @param train_loader : DataLoader for the training dataset
        @param val_loader   : DataLoader for the validation dataset
        @param epochs       : Number of epochs to train the model
        @return             : Dictionary containing training and validation loss and accuracy history"""

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = self.run_epoch(model, train_loader, train=True)
            val_loss, val_accuracy = self.run_epoch(model, val_loader, train=False)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_accuracy)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_accuracy)

            # 4 decimals afer the dot for loss and accuracy
            print(f"Epoch {epoch}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} - "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        return history