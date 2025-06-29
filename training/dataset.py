import torch
from torch.utils.data import Dataset
from typing import Tuple
import pickle
import os

'''
This module defines a custom Text Dataset class for handling text data in PyTorch.

We use character-level tokenization, where each character is treated as a token.
This is far more flexible than word-level tokenization, and it allows for creative text generation.

Note: Subclasses of PyTorch's Dataset class must implement the __len__ and __getitem__ methods.
'''

class TextDataset(Dataset):
    def __init__(self, txt_file_path: str, context_length: int = 128) -> None:
        '''
        Initializes the TextDataset with a file path to a text file and context length.

        Args:
            txt_file_path (str): Path to the text file we'll read from.
            context_length (int): The number of tokens back into our input sequence our model can see/read from.

        Returns:
            None: The dataset is initialized and ready for use.
        '''
        self.context_length = context_length

        # Read the text file
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()

        print(f"Text loaded: {len(self.text)} characters")

        # Create vocabulary (character set)
        self.vocab = sorted(set(self.text))
        self.vocab_size = len(self.vocab)

        # Create mappings between characters and integers
        self.char_to_int = {ch: i for i, ch in enumerate(self.vocab)}
        self.int_to_char = {i: ch for i, ch in enumerate(self.vocab)}

        print(f"Vocabulary size: {self.vocab_size} unique characters")

        # Convert text to integers (representing characters as integers)
        self.data = [self.char_to_int[ch] for ch in self.text]

    def __len__(self) -> int:
        '''
        Returns the total number of training examples in the dataset. 

        Args:
            None

        Returns:
            int: The total number of training examples.

        Note: The length is calculated as the total number of characters minus the context length.
        This ensures that we have enough characters to create a context for each training example.
        '''
        return len(self.data) - self.context_length
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Returns a single training example from the dataset.

        Args:
            idx (int): The index of the training example to retrieve.

        Returns:
            tuple:
                - input (torch.Tensor): The input sequence of integers (context).
                - target (torch.Tensor): The target sequence of integers (including next character).

        For training our GPT model, we need:
            - input: A sequence of integers representing the context (previous characters).
            - target: The same sequence shifted by one character, representing the next character to predict.

        For example, if our text is "hello" and context_length is 4:
            - input: [h, e, l, l] (context)
            - target: [e, l, l, o] (next character to predict)

        This allows the model to learn to predict the next character based on the context.
        '''
        # Get the input sequence (context)
        input = torch.tensor(self.data[idx:idx + self.context_length], dtype=torch.long)
        # Get the target sequence (including next character to predict)
        target = torch.tensor(self.data[idx + 1:idx + self.context_length + 1], dtype=torch.long)

        return input, target
    
    def save_vocab(self, save_path: str) -> None:
        '''
        Saves vocabulary mappings so that we can use them later for inference/text generation.

        Args:
            save_path (str): The path where the vocabulary mappings will be saved.

        Returns:
            None: The vocabulary mappings are saved to the specified path.
        '''
        vocab_data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'char_to_int': self.char_to_int,
            'int_to_char': self.int_to_char
        }

        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(vocab_data, f)

        print(f"Vocabulary saved to {save_path}")

    @staticmethod
    def load_vocab(load_path: str) -> dict:
        '''
        Loads vocabulary data from a saved file.

        Args:
            load_path (str): The path to load the vocabulary mappings from.

        Returns:
            dict: A dictionary containing the vocab, vocab_size, char_to_int, and int_to_char mappings.
        '''
        with open(load_path, 'rb') as f:
            return pickle.load(f)



