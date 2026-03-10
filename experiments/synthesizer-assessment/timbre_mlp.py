import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings

class TimbreMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        save_path: str,
        dropout: float = 0.0,
    ):
        super(TimbreMLP, self).__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.save_path = save_path
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers = []
        prev_size = input_size

        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            prev_size = size
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)
        self.to(self.device)

    def get_params_number(self):
        total_params = sum(p.numel() for p in self.parameters())
        return total_params

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def train_model(
        self,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader = None,
        epochs: int = 500,
        learning_rate: float = 0.001,
        loss_fn=nn.MSELoss(),  # Default: MSE for regression
        optimizer_class=optim.Adam,
        plot_loss: bool = True,
        patience: int = 20,  # Number of epochs to wait before stopping
        lr_scheduler_factor: float = 0.3,  # Factor by which the learning rate will be reduced
        lr_scheduler_patience: int = 5,  # Number of epochs with no improvement after which learning rate will be reduced
    ):

        optimizer = optimizer_class(self.parameters(), lr=learning_rate)
        train_loss_history = [float('inf')]
        valid_loss_history = [float('inf')]
        best_valid_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None
        best_epoch = 0
        lr_history = []  # Track learning rate

        # Initialize learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=lr_scheduler_factor,
            patience=lr_scheduler_patience
        )

        self.train()
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_epoch_loss = 0.0
            for batch_X, batch_y in tqdm(train_dataloader, desc=f"Training - Epoch {epoch+1}/{epochs}, train_loss: {train_loss_history[-1]:.4f}, test_loss: {valid_loss_history[-1]:.4f}"):
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = loss_fn(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
            train_epoch_loss /= len(train_dataloader)
            train_loss_history.append(train_epoch_loss)

            # validing phase
            self.eval()
            valid_epoch_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in valid_dataloader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self(batch_X)
                    loss = loss_fn(outputs, batch_y)
                    valid_epoch_loss += loss.item()
            valid_epoch_loss /= len(valid_dataloader)
            valid_loss_history.append(valid_epoch_loss)

            # Step the learning rate scheduler
            scheduler.step(valid_epoch_loss)
            lr_history.append(optimizer.param_groups[0]['lr'])  # Track learning rate

            # Early stopping and best model saving logic
            if valid_epoch_loss < best_valid_loss:
                best_valid_loss = valid_epoch_loss
                epochs_no_improve = 0
                best_model_state = self.state_dict()  # Save the best model state
                best_epoch = epoch + 1  # Track the best epoch
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                    break

        # Restore the best model state
        if best_model_state is not None:
            self.load_state_dict(best_model_state)

        if plot_loss:
            # Truncate loss history to only include up to the best epoch
            self.plot_loss_curve(
                train_loss_history[:best_epoch],
                valid_loss_history[:best_epoch]
            )
            # Plot learning rate curve
            self.plot_learning_rate_curve(lr_history[:best_epoch])

        # Save the model
        self.save_model()

    def plot_learning_rate_curve(self, lr_history):
        plt.figure(figsize=(10, 5))
        plt.plot(lr_history, label='Learning Rate', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Evolution')
        plt.legend()
        plt.grid(True)
        os.makedirs(os.path.join(self.save_path, "metrics"), exist_ok=True)
        plt.savefig(os.path.join(self.save_path, "metrics", "learning_rate_curve.png"))

    def save_model(self):
        """Save the trained model to the specified path."""
        os.makedirs(self.save_path, exist_ok=True)
        model_path = os.path.join(self.save_path, "timbre_mlp.pth")
        torch.save(self.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def plot_loss_curve(self, train_loss_history: list, valid_loss_history: list):
        """Plot the training loss curve."""
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss_history[1:], label='Training Loss')
        plt.plot(valid_loss_history[1:], label='valid Loss')
        plt.title('Training Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        os.makedirs(os.path.join(self.save_path, "metrics"), exist_ok=True)
        plt.savefig(os.path.join(self.save_path, "metrics", "losses_curves.png"))
        print(f"Loss figure saved to {os.path.join(self.save_path, 'metrics', 'losses_curves.png')}")

    def evaluate_model(
        self,
        eval_dataloader: DataLoader,
        loss_fn=nn.MSELoss(),
        verbose: bool = False
    ) -> tuple[float, torch.Tensor, float]:
        """
        Evaluate the MLP model on validation/test data.

        Args:
            eval_dataloader (DataLoader): DataLoader containing validation/test data.
            loss_fn: Loss function (default: MSELoss).

        Returns:
            tuple[float, torch.Tensor]: Evaluation loss and mean of model outputs.
        """
        self.eval()
        total_loss = 0.0
        all_outputs = []
        total_mae = 0.0

        with torch.no_grad():
            for batch_X, batch_y in eval_dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self(batch_X)
                all_outputs.append(outputs.cpu())
                loss = loss_fn(outputs, batch_y)
                total_mae += torch.mean(torch.abs(outputs - batch_y)).item()
                total_loss += loss.item() * batch_X.size(0)

        # Concatenate all outputs and compute mean
        all_outputs = torch.cat(all_outputs, dim=0)
        # mean_outputs = torch.mean(all_outputs, dim=0).cpu()

        # Return average loss over the entire dataset
        eval_loss = total_loss / len(eval_dataloader.dataset)
        total_mae = total_mae / len(eval_dataloader.dataset)
        if verbose:
            print(f"Eval loss: {eval_loss}")

        return eval_loss, all_outputs, total_mae

    @staticmethod
    def load_model(
        path: str,
        input_size: int,
        hidden_sizes: list,
        output_size: int,
        dropout: float = 0.0,
        verbose: bool = False
    ) -> "TimbreMLP":
        """
        Load a saved model from the specified path.

        Args:
            path (str): Path to the saved model file.
            input_size (int): Size of the input layer.
            hidden_sizes (list): List of hidden layer sizes.
            output_size (int): Size of the output layer.
            dropout (float): Dropout rate (default: 0.0).
            verbose (bool): Whether to print some info (default: True).

        Returns:
            TimbreMLP: An instance of TimbreMLP with loaded weights.
        """
        if verbose:
            warnings.warn("Ensure that the model architecture matches the saved model.")
        model = TimbreMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            save_path=os.path.dirname(path),
            dropout=dropout,
        )
        model.load_state_dict(torch.load(path))
        model.to(model.device)
        return model