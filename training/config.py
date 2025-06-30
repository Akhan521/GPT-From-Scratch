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
    CONTEXT_LENGTH: int = 64     # The number of tokens the model can see/read from (smaller context -> reduced memory use)
    BATCH_SIZE: int = 4          # How many training examples to process in parallel (small batch size for CPU)
    TRAIN_SPLIT: float = 1.0     # Proportion of data to use for training (all data for training, no validation split)

    # Model Architecture (Should match our GPT model's architecture)
    MODEL_DIM: int = 128         # Size of embeddings (balanced size for learning and speed)
    NUM_BLOCKS: int = 4          # Number of transformer blocks (4 is enough for a small model)
    NUM_HEADS: int = 4           # Number of attention heads

    # Training Hyperparameters
    EPOCHS: int = 60             # How many times to go through the entire dataset (enough for training on a small dataset)
    LEARNING_RATE: float = 5e-4  # How fast the model learns (slower learning rate for stability on CPU)
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu' # Device to run the model on (GPU or CPU)

    # Checkpointing configurations
    SAVE_EVERY: int = 20                             # How often to save the model (every 20 epochs)
    CHECKPOINT_DIR: str = 'weights/checkpoints'      # Where to save model weights
    VOCAB_PATH: str = 'vocab/vocab.pkl'              # Path to save/load vocabulary
    FINAL_MODEL_PATH: str = 'weights/final_model.pt' # Path to save the final model weights

    @classmethod
    def print_config(cls) -> None:
        '''Prints all configuration settings.'''
        print("--- Training Configuration ---")
        for attr in dir(cls):
            if not attr.startswith('__') and not callable(getattr(cls, attr)):
                print(f"{attr}: {getattr(cls, attr)}")
        print('-' * 30)

'''
Why These Settings?

# CONTEXT_LENGTH = 64
    - This balances memory usage with context understanding.
    - GPT needs enough context to understand relationships in text and generate coherent responses.
    - The larger the context, the more memory it requires, so we choose a reasonable size.

# BATCH_SIZE = 4
    - This choice of batch size is small enough to fit in memory on a CPU.
    - The larger the batch size, the more stable the gradient updates, but it also requires more memory.
    - It serves as a good starting point for training, allowing the model to learn effectively without running out of memory.

# LEARNING_RATE = 5e-4
    - This smaller learning rate is suitable for training on a CPU, where we want to avoid large updates that could destabilize training.
    - Having a learning rate that is too high can cause the model to diverge, while too low can slow down training.

# EPOCHS = 60
    - This is a reasonable number of epochs to allow the model to learn from the data and see good results.
    - The number of epochs can be adjusted based on the model's performance on validation data.
    - We can stop training early if the model stops improving (early stopping).
'''