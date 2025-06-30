import torch
import torch.nn as nn
import torch.nn.functional as F

'''
This module defines my custom GPT transformer model.

Everything is implemented from scratch, including the transformer architecture.
'''

class GPT(nn.Module):

    class TransformerBlock(nn.Module):

        class MultiHeadSelfAttention(nn.Module):

            class SingleHeadAttention(nn.Module):
                def __init__(self, embed_dim: int, att_dim: int) -> None:
                    '''
                    Initializes a single head of attention.

                    Args:
                        embed_dim (int): Embedding dimension of the model.
                        att_dim (int): Attention dimension of the model.
                        
                    Returns:
                        None: The single attention head is initialized and ready for use.
                    '''
                    super().__init__()
                    # Initialize the linear layers for key, query, and value transformations
                    # It has been found that we get slightly better results when we avoid using bias in these layers.
                    self.key_layer = nn.Linear(embed_dim, att_dim, bias=False)
                    self.query_layer = nn.Linear(embed_dim, att_dim, bias=False)
                    self.value_layer = nn.Linear(embed_dim, att_dim, bias=False)

                    def forward(self, embedded: torch.Tensor) -> torch.Tensor:
                        '''
                        Computes the attention head's output given the input embeddings.

                        Args:
                            embedded (torch.Tensor): Input embeddings of shape (batch_size, context_length, embed_dim).

                        Returns:
                            torch.Tensor: Output of the attention head of shape (batch_size, context_length, att_dim).

                        The formula for computing the attention scores is:
                            Scores = softmax((Q * K^T) / sqrt(d_k)) * V, 
                            where Q is the query tensor, K is the key tensor, d_k is the attention dimension, and V is the value tensor.
                        '''
                        # Compute the key, query, and value tensors
                        k = self.key_layer(embedded)   # Shape: (batch_size, context_length, att_dim)
                        q = self.query_layer(embedded) # Shape: (batch_size, context_length, att_dim)
                        v = self.value_layer(embedded) # Shape: (batch_size, context_length, att_dim)

                        # Compute the attention scores
                        _, context_length, att_dim = k.shape
                        scores = q @ torch.transpose(k, 1, 2) / (att_dim ** 0.5)  # Shape: (batch_size, context_length, context_length)

                        # To prevent our model from attending to/seeing future tokens, we apply a mask to the scores.
                        pre_mask = torch.tril(torch.ones(context_length, context_length)).to(embedded.device)  # Lower triangular mask
                        mask = (pre_mask == 0).to(embedded.device)
                        # We'll be using the softmax function, so setting the masked positions to -inf will ensure they get a probability of 0.
                        scores = scores.masked_fill(mask, float('-inf'))

                        # Apply softmax to get attention weights
                        scores = F.softmax(scores, dim=-1)  # Shape: (batch_size, context_length, context_length)

                        # Compute the output of the attention head
                        return scores @ v  # Shape: (batch_size, context_length, att_dim)
                    
            def __init__(self, model_dim: int, num_heads: int) -> None:
                '''
                Initializes the multi-head self-attention mechanism.

                Args:
                    model_dim (int): The model dimension (embedding dimension).
                    num_heads (int): The number of attention heads to use.

                Returns:
                    None: The multi-head self-attention mechanism is initialized and ready for use.
                '''
                super().__init__()
                # Ensure that the model dimension is divisible by the number of heads
                assert model_dim % num_heads == 0, "Multi-Head Attention: Model dimension must be divisible by number of heads."

                # Initialize our list of single attention heads.
                self.attention_heads = nn.ModuleList([
                    self.SingleHeadAttention(model_dim, model_dim // num_heads)
                    for _ in range(num_heads)
                ])

                # Initialize the linear layer to combine the outputs of all attention heads
                # This layer will take the concatenated output of all heads and project it back to the model dimension.
                self.compute_output = nn.Linear(model_dim, model_dim)

                # Initialize a dropout layer to apply dropout to the attention output
                self.dropout = nn.Dropout(0.2) # Using p = 0.2 for dropout rate (20% likelihood of dropping out a unit)

            def forward(self, embedded: torch.Tensor) -> torch.Tensor:
                '''
                Computes the multi-head self-attention output given the input embeddings.

                Args:
                    embedded (torch.Tensor): Input embeddings of shape (batch_size, context_length, model_dim).

                Returns:
                    torch.Tensor: Output of the multi-head self-attention mechanism of shape (batch_size, context_length, model_dim).
                '''
                # Compute the outputs of all attention heads. Each head returns a tensor of shape (batch_size, context_length, att_dim).
                attention_outputs = [head(embedded) for head in self.attention_heads] 

                # Concatenate the outputs of all attention heads along the last dimension -> Shape: (batch_size, context_length, model_dim)
                attention_outputs = torch.cat(attention_outputs, dim=-1)

                # Combine the outputs of all attention heads and apply dropout
                return self.dropout(self.compute_output(attention_outputs))  # Shape: (batch_size, context_length, model_dim)
            
        class FeedForwardNeuralNetwork(nn.Module):
            def __init__(self, model_dim: int) -> None:
                '''
                Initializes the vanilla feed-forward neural network (FFNN) used in the transformer block.

                Args:
                    model_dim (int): The model dimension (embedding dimension).

                Returns:
                    None: The FFNN is initialized and ready for use.
                '''
                super().__init__()
                self.first_linear_layer = nn.Linear(model_dim, model_dim * 4)  # First linear layer expands the dimension
                self.relu = nn.ReLU()                                          # ReLU activation function
                self.second_linear_layer = nn.Linear(model_dim * 4, model_dim) # Second linear layer reduces the dimension back to model_dim
                self.dropout = nn.Dropout(0.2)                                 # Dropout layer to prevent overfitting

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                '''
                Computes the output of the feed-forward neural network given the input tensor.

                Args:
                    x (torch.Tensor): Input tensor of shape (batch_size, context_length, model_dim).

                Returns:
                    torch.Tensor: Output tensor of shape (batch_size, context_length, model_dim).
                '''
                return self.dropout(self.second_linear_layer(self.relu(self.first_linear_layer(x))))  # Apply FFNN with dropout
            
        def __init__(self, model_dim: int, num_heads: int) -> None:
            '''
            Initializes a single transformer block which consists of multi-head self-attention and a feed-forward neural network.

            Args:
                model_dim (int): The model dimension (embedding/attention dimension).
                num_heads (int): The number of attention heads to use.

            Returns:
                None: The transformer block is initialized and ready for use.
            '''
            super().__init__()
            # Initialize the multi-head self-attention mechanism
            self.mhsa = self.MultiHeadSelfAttention(model_dim, num_heads)

            # Initialize the feed-forward neural network
            self.ffnn = self.FeedForwardNeuralNetwork(model_dim)

            # Initialize normalization layers for both attention and feed-forward outputs
            self.first_norm_layer = nn.LayerNorm(model_dim)
            self.second_norm_layer = nn.LayerNorm(model_dim)

        def forward(self, embedded: torch.Tensor) -> torch.Tensor:
            '''
            Computes the output of the transformer block given the input embeddings.

            Args:
                embedded (torch.Tensor): Input embeddings of shape (batch_size, context_length, model_dim).

            Returns:
                torch.Tensor: Output of the transformer block of shape (batch_size, context_length, model_dim).
            '''
            # Compute the output of the multi-head self-attention mechanism
            embedded = embedded + self.mhsa(self.first_norm_layer(embedded))  # Add skip/residual connection and apply normalization
            embedded = embedded + self.ffnn(self.second_norm_layer(embedded)) # Add another skip/residual connection and apply normalization
            return embedded  # Shape: (batch_size, context_length, model_dim)

    def __init__(self, vocab_size: int, context_length: int, model_dim: int, num_blocks: int, num_heads: int) -> None:
        '''
        Initializes the GPT model with the given parameters.

        Args:
            vocab_size (int): Size of the vocabulary.
            context_length (int): Length of the context (input sequence) for the model.
            model_dim (int): Attention and embedding dimension of the model.
            num_blocks (int): Number of transformer blocks (layers) in the model.
            num_heads (int): Number of attention heads in each transformer block.

        Returns:
            None: The model is initialized and ready for use.
        '''
        super().__init__()
        # Initialize the token and positional embedding layers to convert input tokens and positions into embeddings.
        self.token_embeddings = nn.Embedding(vocab_size, model_dim)
        self.pos_embeddings = nn.Embedding(context_length, model_dim)

        # Initialize the transformer blocks
        self.transformer_blocks = nn.Sequential()
        for _ in range(num_blocks):
            self.transformer_blocks.append(self.TransformerBlock(model_dim, num_heads))

        # Initialize the final normalization layer
        self.final_norm_layer = nn.LayerNorm(model_dim)

        # Initialize the final linear layer to project the output back to the vocabulary size
        self.vocab_projection = nn.Linear(model_dim, vocab_size)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        '''
        Computes the output of the GPT model given the input context.

        Args:
            context (torch.Tensor): Input tensor of shape (batch_size, context_length) containing token indices.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, context_length, vocab_size) containing logits for each token in the vocabulary.
        '''
        # Get the context length 
        _, context_length = context.shape

        # Get the token embeddings for the input context
        token_embeddings = self.token_embeddings(context)

        # Get the positional embeddings for the input context
        pos_embeddings = self.pos_embeddings(torch.arange(context_length, device=context.device))

        # Combine token and positional embeddings
        embedded = token_embeddings + pos_embeddings  # Shape: (batch_size, context_length, model_dim)

        # Compute the output of the GPT model (raw output/logits for each token in the vocabulary)
        return self.vocab_projection(self.final_norm_layer(self.transformer_blocks(embedded))) # Shape: (batch_size, context_length, vocab_size)
