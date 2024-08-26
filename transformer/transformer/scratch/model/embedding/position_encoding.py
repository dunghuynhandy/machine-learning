
import torch
import math
class PositionEncoding(torch.nn.Module):
    """
    Position encoding in a transformer provides information about the order of
    tokens in a sequence, since the model itself doesn't inherently capture
    positional information. It adds or concatenates these positional encodings
    to the input embeddings, allowing the model to differentiate between
    tokens' positions. This enables the transformer to understand the sequence
    structure, which is crucial for tasks like translation and text generation.

    Args:
        d_model (int): the number of expected features in the encoder/decoder inputs
        max_len (int): max length of source/target input
        device (str): cuda or cpu
        
    """
    def __init__(self,
                 d_model,
                 max_len):
        super(PositionEncoding, self).__init__()
        # Create a matrix of shape (max_len, d_model) filled with zeros
        pe = torch.zeros(max_len, d_model)

        # Create a column vector [0, 1, 2, ..., max_len-1].unsqueeze(1) makes it a column vector
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Compute the divisor for the sine and cosine functions, with odd and even indexing
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices in the array (2i)
        pe[:, 0::2] = torch.sin(position * div_term.unsqueeze(0))

        # Apply cosine to odd indices in the array (2i+1)
        pe[:, 1::2] = torch.cos(position * div_term.unsqueeze(0))

        # Add a batch dimension
        pe = pe.unsqueeze(0).transpose(0, 1)

        # Register the positional encoding as a buffer, so it's not trained but part of the model
        self.register_buffer('pe', pe)
    def forward(self, x):
        # Add the positional encoding to the input embeddings
        x = x + self.pe[:x.size(0), :]
        return x




