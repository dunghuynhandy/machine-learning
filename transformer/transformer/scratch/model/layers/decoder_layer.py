import torch
from model.layers import MultiHeadAttention
from model.layers.norm import LayerNorm
from model.layers.feed_forward import FeedForward
from loguru import logger
class DecoderLayer(torch.nn.Module):

    
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
            dim_feedforward (int):  the number of expected features in the encoder outputs
            n_head (int): number of multi-head attention
            drop_prob (float): dropdown probability
        """
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head = n_head)
        
        self.drop_1 = torch.nn.Dropout(drop_prob)
        self.norm_1 =LayerNorm(d_model=d_model)
        
        
        self.cross_attention = MultiHeadAttention(d_model=d_model, n_head = n_head)
        self.ffn = FeedForward(d_model=d_model, dim_feedforward=dim_feedforward)
        self.drop_2 = torch.nn.Dropout(drop_prob)
        self.norm_2 =LayerNorm(d_model=d_model)
        
        self.drop_3 = torch.nn.Dropout(drop_prob)
        self.norm_3 =LayerNorm(d_model=d_model)
        
        
    def forward(self, x, y, x_mask, y_mask):
        logger.debug(x.size())
        y_ = y
        y = self.self_attention(q=y, k=y, v=y, mask= y_mask)
        y = self.drop_1(y)
        y = y + y_
        y = self.norm_1(y)
        logger.debug(y.size())
        
        if x is not None:
            y_ = y
            y = self.cross_attention(q=y, k=x, v=x, mask= x_mask)
            y = self.drop_2(y)
            y = y + y_
            y = self.norm_2(y)
        
        y_ = y
        y = self.drop_3(y)
        y = y + y_
        y = self.norm_3(y)
        return y
        
        
    