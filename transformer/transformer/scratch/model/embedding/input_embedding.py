from torch import nn

class InputEmbedding(nn.Embedding):
    """_summary_
    
    Input embedding in a transformer converts tokens (words or subwords) 
    into dense vectors of a fixed size, which the model can process more 
    effectively. This embedding captures semantic meaning, enabling the
    model to understand relationships between words in the input sequence. 
    It also standardizes input size for further processing by the transformer's 
    layers.
    Args:
        vocal_size (int): vocabulary size of source and target languages
        d_model (int):  the number of expected features in the encoder/decoder inputs 
    """
    def __init__(self,
                 vocal_size,
                 d_model):
        super(InputEmbedding, self).__init__(vocal_size, d_model)
    
    