import torch
import torch.nn as nn
import torch.nn.functional as F
from model.gpt import GPT

'''This module handles text generation using our trained model.'''

def generate_text(model: GPT, vocab_data: dict, device: torch.device, prompt: str = '', max_length: int = 500, temperature: float = 1.0) -> str:
    '''
    Generates text using the trained GPT model.

    Args:
        model (GPT): The trained GPT model.
        vocab_data (dict): A dictionary containing vocabulary data.
        device (torch.device): The device to run the model on (CPU or GPU).
        prompt (str): The initial text prompt to start generation from.
        max_length (int): The number of tokens to generate.
        temperature (float): Controls randomness in predictions. Higher values mean more random outputs.

    Returns:
        str: The generated text.

    Understanding Temperature:
        - Temperature = 1.0: Normal sampling, balanced randomness.
        - Temperature < 1.0: Less random, more focused on high-probability tokens.
        - Temperature > 1.0: More random, allows for exploration of less probable tokens.

    How Text Generation Works:
        1. Start with the prompt (or an empty string) and convert it to tokens.
        2. Use the model to predict the next token based on the current sequence.
        3. Sample the next token based on the model's output probabilities adjusted by temperature.
        4. Append the predicted token to the sequence and repeat until we generate the desired length.
    '''
    # Ensure the model is in evaluation mode (disables dropout, etc.)
    model.eval()

    # Get the vocabulary mapping
    char_to_int = vocab_data['char_to_int']
    int_to_char = vocab_data['int_to_char']

    # Convert the prompt to a list of tokens (integers)
    if prompt:
        context = [char_to_int.get(char, 0) for char in prompt]
    else:
        # Start with a newline character (or first character in the vocabulary))
        context = [0] 

    # Convert context to a tensor and add a batch dimension
    context = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)

    generated_text = []

    # Generate tokens (chars) one-by-one until max_length is reached
    with torch.no_grad(): 
        for _ in range(max_length):
            # Limit the input context to the model's context length. Our context has shape (B, C), where B = 1 (batch size) and C = context length.
            model_context_length = model.pos_embeddings.num_embeddings # This is the maximum context length the model can handle.
            if len(context.T) > model_context_length:
                context = context[:, -model_context_length:]

            # Get model predictions
            outputs = model(context) # Shape: (B, C, V), where B = 1 (batch size), C = context length, V = vocab_size

            # We're only interested in the last position's predictions (to predict the next token)
            logits = outputs[:, -1, :]

            # Apply temperature to control randomness
            if temperature != 1.0:
                logits = logits / temperature

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token/char from the probability distribution
            # Multinomial sampling: higher probabilities = more likely to be chosen
            next_char = torch.multinomial(probs, num_samples=1)

            # Add the predicted char to our context for the next iteration
            # Context has shape (B, C), we need to append the new char along the last dimension
            context = torch.cat((context, next_char), dim=-1)

            # Convert the predicted token (int) back to a character and append to generated text
            generated_text.append(int_to_char[next_char.item()])

    # Join the generated characters into a single string
    return ''.join(generated_text)

'''
For a better understanding of how temperature affects text generation:
    - As t -> 0, the model becomes deterministic, always picking the highest probability token (greedy sampling).
    - If t = 1, the model samples from the distribution as is (normal random sampling).
    - If t > 1, the model becomes more exploratory, allowing less probable tokens to be chosen, leading to more diverse outputs.
      This can be useful for creative text generation, but may also lead to nonsensical outputs.
'''
