'''
This will be our text generation script for the trained GPT model.

Note: once you have trained your model, you can use this script to generate text. 
    - If you need to train your model, use train.py instead.

This script will:
    1. Load the trained GPT model.
    2. Load the vocabulary mappings.
    3. Generate text based on a prompt.
'''
import torch
from model.gpt import GPT
from utils.text_generation import generate_text
from training.config import TrainingConfig
from training.dataset import TextDataset
import os

def load_model(model_path: str, vocab_data: dict, config: TrainingConfig) -> GPT:
    '''
    Loads the trained GPT model from the specified path.

    Args:
        model_path (str): Path to the saved GPT model.
        config (TrainingConfig): Our model's training configuration.

    Returns:
        GPT: The loaded GPT model.
    '''
    # Check if the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f'Model weights not found at {model_path}. Please ensure you have trained the model first using train.py.'
        )
    
    # Create the GPT model instance
    model = GPT(
        vocab_size=vocab_data['vocab_size'],
        context_length=config.CONTEXT_LENGTH,
        model_dim=config.MODEL_DIM,
        num_blocks=config.NUM_BLOCKS,
        num_heads=config.NUM_HEADS,
    )

    # Load the model weights
    device = torch.device(config.DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)  # Move the model to the specified device (CPU or GPU)
    model.eval()      # Set the model to evaluation mode

    print(f'\nModel loaded successfully from {model_path}\n')
    return model

def generate(model: GPT, vocab_data: dict, prompt: str, max_length: int, temperature: float, config: TrainingConfig) -> None:
    '''
    Generates text using the trained GPT model.

    Args:
        model (GPT): The trained GPT model.
        vocab_data (dict): The vocabulary data containing mappings.
        prompt (str): The initial text prompt to start the generation.
        max_length (int): The number of tokens to generate.
        temperature (float): Controls the randomness of the generation (higher = more random).
        config (TrainingConfig): Our model's training configuration.

    Returns:
        None: The function prints the generated text to the console.
    '''
    print('Generating text...')
    print(f'\n\tPrompt: {prompt}')
    print(f'\n\tMax Length: {max_length} characters to generate.')
    print(f'\n\tTemperature: {temperature} (controls randomness)\n')
    print('=' * 60)

    # Generate text using the model
    generated_text = generate_text(
        model=model,
        vocab_data=vocab_data,
        device=torch.device(config.DEVICE),
        prompt=prompt,
        max_length=max_length,
        temperature=temperature
    )

    print(f'\nGenerated text:\n{prompt}{generated_text}\n')

def generate_text_pipeline() -> None:
    '''
    Runs the text generation pipeline for the trained GPT model.

    This function orchestrates the entire text-generation workflow:
    1. Loads the vocabulary data.
    2. Loads the trained GPT model.
    3. Generates text based on a user-defined prompt.
    '''
    print('\nStarting text generation pipeline...')
    print('=' * 60)

    try:
        # Load the training configuration
        config = TrainingConfig()

        # Check if the vocabulary file exists
        if not os.path.exists(config.VOCAB_PATH):
            raise FileNotFoundError(
                f'Vocabulary file not found at {config.VOCAB_PATH}. Please ensure you have trained the model first using train.py.'
            )

        # Load vocabulary data
        vocab_data = TextDataset.load_vocab(config.VOCAB_PATH)

        # Load the trained model
        model = load_model(config.FINAL_MODEL_PATH, vocab_data, config)

        # Define parameters for text generation
        prompt = 'Here is Edward Bear, coming'
        max_length = 500   # Number of tokens to generate
        temperature = 1.0  # Controls randomness in generation

        # Generate text
        generate(model, vocab_data, prompt, max_length, temperature, config)

        print('=' * 60)
        print('\nText generation completed successfully!\n')

    except FileNotFoundError as e:
        print(f'Error: {str(e)}')
        print('Please ensure you have trained the model first using train.py.\n')

    except Exception as e:
        print(f'An error occurred during text generation: {str(e)}')
        print('Please check your model files and try again.\n')

if __name__ == '__main__':
    generate_text_pipeline()