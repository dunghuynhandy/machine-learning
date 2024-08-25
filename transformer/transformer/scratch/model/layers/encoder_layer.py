import torch
from model.layers import MultiHeadAttention
from model.layers.norm import LayerNorm
from model.layers.feed_forward import FeedForward

class EncoderLayer(torch.nn.Module):

    
    def __init__(self, d_model, dim_feedforward, n_head, drop_prob):
        """

        An encoder layer in a transformer model is a key component that processes input sequences to 
        generate meaningful representations. It consists of two main sub-layers: a multi-head 
        self-attention mechanism, which allows the model to focus on different parts of the input sequence, 
        and a feedforward neural network, which further processes the information. Each sub-layer 
        is followed by layer normalization and residual connections, ensuring stable training and 
        allowing the model to retain essential features from the input. Multiple encoder layers are stacked 
        to create deep, hierarchical representations of the input data, enabling the model to capture complex
        patterns and dependencies.
        
        Args:
            d_model (int):  the number of expected features in the encoder/decoder inputs
            ffn_hidden (int):  the number of expected features in the encoder outputs
            n_head (int): number of multi-head attention
            drop_prob (float): dropdown probability
        """
        super(EncoderLayer, self).__init__()
        self.self_attention  = MultiHeadAttention(d_model=d_model, n_head = n_head)
        
        self.drop_1 = torch.nn.Dropout(drop_prob)
        self.norm_1 =LayerNorm(d_model=d_model)
        
        self.ffn = FeedForward(d_model=d_model, dim_feedforward=dim_feedforward)
        self.drop_2 = torch.nn.Dropout(drop_prob)
        self.norm_2 =LayerNorm(d_model=d_model)
        
        
    def forward(self, x, mask):
        x_ = x
        x = self.self_attention(q=x, k=x, v=x, mask= mask)
        x = self.drop_1(x)
        x = x + x_
        x = self.norm_1(x)
        
        x_ = x
        x = self.ffn(x)
        x = self.drop_2(x)
        x = x + x_
        x = self.norm_2(x)
        
        return x
        
        
    