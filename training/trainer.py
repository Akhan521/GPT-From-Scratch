import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from training.config import TrainingConfig
from model.gpt import GPT
import os

'''
This class handles the training process for a GPT model.

We separate the training logic from the model for the following reasons:
1. To keep our code modular and maintainable. The model handles the forward pass and architecture,
   while the trainer handles all training logic.
2. To build a reusable training framework that can be applied to different models or tasks.
3. To allow for easier testing and debugging of the training process without modifying the model code.
'''
class GPTTrainer:
    def __init__(self, model: GPT, config: TrainingConfig) -> None:
        '''
        Initializes the GPTTrainer with a model and training configuration.

        Args:
            model (GPT): The GPT model to be trained.
            config (TrainingConfig): Configuration object containing training parameters.

        Returns:
            None: The trainer is initialized with the model and configuration.
        '''
        self.model = model.to(config.DEVICE)
        self.config = config
        self.device = config.DEVICE

        # To track training/validation loss
        self.train_losses = []
        self.val_losses = []

        # Optimizer and loss function
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()

        print("Trainer initialized on device: ", self.device)

    def train_epoch(self, train_loader: DataLoader) -> float:
        '''
        Trains the model for one epoch (one full pass through the training data).

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.

        Returns:
            float: Average training loss for the epoch.
        '''
        # Set the model to training mode (enables dropout, batchnorm, etc.)
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Move data to the appropriate device (GPU or CPU)
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Clear gradients from the previous step
            self.optimizer.zero_grad()

            # Forward pass: get model predictions
            outputs = self.model(inputs) # Shape: (batch_size, context_length, vocab_size)

            # Reshape outputs and targets for loss calculation
            # CrossEntropyLoss expects inputs of shape (N, V) and targets of shape (N, ) where N = batch_size * context_length and V = vocab_size
            _, _, vocab_size = outputs.shape
            outputs_reshaped = outputs.view(-1, vocab_size)  # Shape: (batch_size * context_length, vocab_size)
            targets_reshaped = targets.view(-1)              # Shape: (batch_size * context_length, )

            # Calculate loss
            loss = self.criterion(outputs_reshaped, targets_reshaped)

            # Backward pass: compute gradients
            loss.backward()

            # Update model parameters
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # Calculate average loss for the epoch
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        '''
        Evaluates the model on the validation dataset.
        
        Args:
            val_loader (DataLoader): DataLoader for the validation dataset.
            
        Returns:
            float: Average validation loss.

        Why is validation important?
            - It helps us monitor the model's performance on unseen data. Is our model overfitting?
            - It helps us know when to stop training (early stopping).
        '''
        # Set the model to evaluation mode (disables dropout, batchnorm, etc.)
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        # There is no need to compute gradients during validation (saves memory and computation)
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to the appropriate device (GPU or CPU)
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass: get model predictions
                outputs = self.model(inputs)

                # Reshape outputs and targets for loss calculation (same as in training)
                _, _, vocab_size = outputs.shape
                outputs_reshaped = outputs.view(-1, vocab_size)  # Shape: (batch_size * context_length, vocab_size)
                targets_reshaped = targets.view(-1)              # Shape: (batch_size * context_length, )

                # Calculate loss
                loss = self.criterion(outputs_reshaped, targets_reshaped)
                total_loss += loss.item()
                num_batches += 1

        # Calculate average loss for the validation set
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int) -> None:
        '''
        Our full training loop that runs for multiple epochs and saves the model periodically.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            epochs (int): Number of epochs to train the model.

        Returns:
            None: The model is trained and saved periodically.
        '''
        print(f"Starting training for {epochs} epochs...")
        self.config.print_config()

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            # Training step
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            # Validation step
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                print(f"Train Loss: {train_loss:.4f}")

            # Save model periodically
            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_checkpoint(epoch + 1)

            # Early stopping check (optional)
            if self.should_stop_early():
                print("Early stopping triggered. Stopping training.")
                break

        # Save the final model at the end of training
        self.save_final_model()
        print("Training complete!")

    def save_checkpoint(self, epoch: int) -> None:
        '''
        Saves the model's state dictionary and optimizer state for checkpointing.

        Args:
            epoch (int): The current epoch number, used in the filename.

        Returns:
            None: The model checkpoint is saved to the specified directory.
        '''
        # Ensure the checkpoint directory exists
        os.makedirs(self.config.CHECKPOINT_DIR, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config
        }

        checkpoint_path = os.path.join(self.config.CHECKPOINT_DIR, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")


    def save_final_model(self) -> None:
        '''
        Saves the final trained model to the specified path.

        Args:
            None

        Returns:
            None: The final model weights are saved to the specified path.
        '''
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.config.FINAL_MODEL_PATH), exist_ok=True)
        torch.save(self.model.state_dict(), self.config.FINAL_MODEL_PATH)
        print(f"Final model saved at {self.config.FINAL_MODEL_PATH}")

    def should_stop_early(self, patience: int = 5) -> bool:
        '''
        Checks if early stopping should be triggered based on validation loss.

        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.

        Returns:
            bool: True if early stopping should be triggered, False otherwise.

        If validation loss has not improved for 'patience' epochs, we stop training.
        '''
        # If we don't have enough validation losses to check or we don't have validation at all, don't stop early
        if not self.val_losses or len(self.val_losses) < patience + 1:
            return False
        
        # Check if validation loss hasn't improved for 'patience' epochs
        recent_losses = self.val_losses[-patience:]
        best_loss = min(recent_losses)

        # If the current loss isn't the best in the last 'patience' epochs, we stop early
        return self.val_losses[-1] > best_loss
    
    def plot_losses(self) -> None:
        '''
        Plots the training and validation losses over epochs.

        Args:
            None

        Returns:
            None: Displays the loss plots.
        '''
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', color='blue')

        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', color='orange')

        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Save the plot as an image file
        plt.savefig('Training_Progress.png')
        print("Loss plot saved as 'Training_Progress.png'")

