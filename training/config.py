import torch

'''
This module defines the configuration for training a GPT model.

Why is this beneficial?
    - It centralizes all training-related configurations, making it easier to manage and modify.
    - It makes hyperparameter experimentation easier and more organized.
    - It prevents having values scattered throughout the codebase, which can lead to inconsistencies.
'''

class TrainingConfig:
    '''Configuration class for training a GPT model.'''

    # Data Configurations
    CONTEXT_LENGTH: int = 128    # The number of tokens the model can see/read from
    BATCH_SIZE: int = 32         # How many training examples to process in parallel
    TRAIN_SPLIT: float = 0.9     # Proportion of data to use for training (0.9 = 90% training, 10% validation)

    # Model Architecture (Should match our GPT model's architecture)
    MODEL_DIM: int = 252         # Size of embeddings
    NUM_BLOCKS: int = 6          # Number of transformer blocks
    NUM_HEADS: int = 6           # Number of attention heads

    # Training Hyperparameters
    EPOCHS: int = 50             # How many times to go through the entire dataset
    LEARNING_RATE: float = 1e-3  # How fast the model learns
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu' # Device to run the model on (GPU or CPU)

    # Checkpointing configurations
    SAVE_EVERY: int = 10                             # How often to save the model (every 10 epochs)
    CHECKPOINT_DIR: str = 'weights/checkpoints'      # Where to save model weights
    VOCAB_PATH: str = 'vocab/vocab.pkl'              # Path to save/load vocabulary
    FINAL_MODEL_PATH: str = 'weights/final_model.pt' # Path to save the final model weights

    # Validation configurations
    EVAL_EVERY: int = 1          # How often to evaluate the model on validation data (every epoch)

    @classmethod
    def print_config(cls):
        '''Prints all configuration settings.'''
        print("--- Training Configuration ---")
        for attr in dir(cls):
            if not attr.startswith('__') and not callable(getattr(cls, attr)):
                print(f"{attr}: {getattr(cls, attr)}")
        print('-' * 30)

'''
Why These Settings?

# CONTEXT_LENGTH = 128
    - This balances memory usage with context understanding.
    - GPT needs enough context to understand relationships in text and generate coherent responses.
    - The larger the context, the more memory it requires, so we choose a reasonable size.

# BATCH_SIZE = 32
    - This choice of batch size fits well with most GPUs.
    - The larger the batch size, the more stable the gradient updates, but it also requires more memory.
    - It serves as a good starting point for training, allowing the model to learn effectively without running out of memory.

# LEARNING_RATE = 1e-3
    - This is a common starting point for the ADAM optimizer.
    - Having a learning rate that is too high can cause the model to diverge, while too low can slow down training.

# EPOCHS = 50
    - This is a reasonable number of epochs to allow the model to learn from the data and see good results.
    - The number of epochs can be adjusted based on the model's performance on validation data.
    - We can stop training early if the model stops improving (early stopping).
'''