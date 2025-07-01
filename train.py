'''
This will be our main training script for the GPT model.

For training, ensure the text you want to train on is stored in data/training_text.txt.
'''
import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple
import os

# Our custom modules
from training.dataset import TextDataset
from training.trainer import GPTTrainer
from training.config import TrainingConfig
from model.gpt import GPT

def prepare_data(config: TrainingConfig) -> Tuple[DataLoader, DataLoader, int]:
    '''
    Prepares our datasets and dataloaders.

    Args: 
        config (TrainingConfig): Our model's training configuration.

    Returns:
        tuple:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - vocab_size (int): Size of the vocabulary used in the dataset.
    '''
    print('\nPreparing datasets...')

    filepath = 'data/training_text.txt'
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training text file not found at {filepath}")
    
    # Create the dataset
    dataset = TextDataset(txt_file_path=filepath, context_length=config.CONTEXT_LENGTH)

    # Save our vocab for later use (e.g., in text generation)
    os.makedirs(os.path.dirname(config.VOCAB_PATH), exist_ok=True)
    dataset.save_vocab(config.VOCAB_PATH)

    # Train/Validation Split
    train_size = int(len(dataset) * config.TRAIN_SPLIT)
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(
        dataset=dataset,
        lengths=[train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )

    print(f'Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples')

    # Create DataLoaders for both training and validation sets
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # Shuffle training data for better generalization
        num_workers=2, # Number of subprocesses to use for data loading
        pin_memory=True if config.DEVICE == 'cuda' else False # For speeding up data transfer to GPU
    )

    # If there are no validation samples, we can skip creating a DataLoader for validation.
    if val_size == 0:
        print('No validation samples available. Skipping validation DataLoader creation.')
        return train_loader, None, dataset.vocab_size
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # No need to shuffle validation data
        num_workers=2,
        pin_memory=True if config.DEVICE == 'cuda' else False
    )

    return train_loader, val_loader, dataset.vocab_size

def create_gpt_model(vocab_size: int, config: TrainingConfig) -> GPT:
    '''
    Creates and initializes our GPT model.

    Args:
        vocab_size (int): Size of the vocabulary used in the dataset.
        config (TrainingConfig): Our model's training configuration.

    Returns:
        gpt_model (GPT): An instance of the GPT model initialized with the given configuration.
    '''
    print('Creating GPT model...')
    gpt_model = GPT(
        vocab_size=vocab_size,
        context_length=config.CONTEXT_LENGTH,
        model_dim=config.MODEL_DIM,
        num_blocks=config.NUM_BLOCKS,
        num_heads=config.NUM_HEADS
    )

    # Print model summary
    total_params = sum(p.numel() for p in gpt_model.parameters())
    trainable_params = sum(p.numel() for p in gpt_model.parameters() if p.requires_grad)
    print(f'GPT Model created with {total_params} total parameters, {trainable_params} trainable parameters.\n')

    return gpt_model

def train_model(gpt_model: GPT, train_loader: DataLoader, val_loader: DataLoader, config: TrainingConfig) -> GPTTrainer:
    '''
    Trains the GPT model using the provided training and validation data loaders.

    Args:
        gpt_model (GPT): The GPT model to be trained.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        config (TrainingConfig): Our model's training configuration.

    Returns:
        trainer (GPTTrainer): An instance of the GPTTrainer that manages the training process.
    '''
    # Create the trainer
    trainer = GPTTrainer(model=gpt_model, config=config)

    try:
        # Run the training loop
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.EPOCHS
        )

        print('Training completed successfully.\n')
        return trainer
    
    except KeyboardInterrupt:
        print('\nTraining interrupted (ctrl + c). Saving current progress...')
        trainer.save_checkpoint(epoch=len(trainer.train_losses))
        print('Checkpoint saved. You may resume training later.\n')
        return trainer
    
    except Exception as e:
        print(f'\nTraining failed with error: {str(e)}')
        print('Saving current progress...\n')
        trainer.save_checkpoint(epoch=len(trainer.train_losses))
        raise e

def save_training_summary(trainer: GPTTrainer, config: TrainingConfig, vocab_size: int) -> str:
    '''
    Saves a summary of the training process and plots.

    Args:
        trainer (GPTTrainer): The trainer instance containing training history.
        config (TrainingConfig): Our model's training configuration.
        vocab_size (int): Size of the vocabulary used in the dataset.

    Returns:
        str: Our training summary as a string.
    '''
    print('\nSaving training summary and plots...')

    # Plot training and validation losses
    trainer.plot_losses()

    # Our training summary
    summary = f'''
    --- GPT Training Summary ---
    Final Training Loss: {trainer.train_losses[-1]:.4f}
    Final Validation Loss: {trainer.val_losses[-1]:.4f if trainer.val_losses else 'N/A'}
    Total Training Epochs: {len(trainer.train_losses)}
    Vocabulary Size: {vocab_size}

    Model Configuration:
    Context Length: {config.CONTEXT_LENGTH}
    Model Dimension: {config.MODEL_DIM}
    Number of Blocks: {config.NUM_BLOCKS}
    Number of Heads: {config.NUM_HEADS}
    Batch Size: {config.BATCH_SIZE}
    Learning Rate: {config.LEARNING_RATE}

    Files Created:
        - Model: {config.FINAL_MODEL_PATH}
        - Vocabulary: {config.VOCAB_PATH}
        - Checkpoints: {config.CHECKPOINT_DIR}/

    '''

    # Save the summary to a text file
    summary_path = 'training_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)

    print(f'Training summary saved to {summary_path}\n')
    return summary

def run_training_pipeline() -> None:
    '''
    Runs the entire training pipeline for the GPT model.

    This function orchestrates the entire workflow:
    1. Prepares the dataset and dataloaders.
    2. Creates the GPT model.
    3. Trains the model.
    4. Saves the training summary.
    '''
    print('\nStarting GPT training pipeline...')

    # Initialize the training configuration
    config = TrainingConfig()

    try:
        # Step 1: Prepare the dataset and dataloaders
        train_loader, val_loader, vocab_size = prepare_data(config)

        # Step 2: Create the GPT model
        gpt_model = create_gpt_model(vocab_size, config)

        # Step 3: Train the model
        trainer = train_model(gpt_model, train_loader, val_loader, config)

        # Step 4: Save the training summary
        summary = save_training_summary(trainer, config, vocab_size)
        print(summary)

        print('\nTraining completed successfully!')
        print(f'Your trained model is saved at {config.FINAL_MODEL_PATH}')
        print('You can now use generate.py to generate text with your trained model!\n')

    except Exception as e:
        print(f'Training failed with error: {str(e)}')
        print('Please check the error message above for details.\n')
        raise e
    
if __name__ == '__main__':
    run_training_pipeline()
