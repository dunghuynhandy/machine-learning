import torch
from torch import nn
from loguru import logger
class ScaledDotProductAttention(torch.nn.Module):
    """_summary_
        
    Scaled Dot-Product Attention allows a transformer to focus on relevant parts of an input sequence 
    by computing attention scores between tokens. It achieves this by taking the dot product of queries
    and keys, scaling the result to prevent large gradients, and applying a softmax function to obtain
    attention weights. These weights are then used to create a weighted sum of the values, determining 
    the output for each token. The scaling step ensures stability during training, particularly when 
    dealing with high-dimensional data. This mechanism is essential for capturing contextual relationships 
    in sequences, enabling transformers to effectively understand and process complex language patterns
    Args:
    """
    
    def __init__(self,):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, q, k, v, mask):
        """_summary_

        Args:
            q (_type_): Query matrix (shape: [batch_size, num_heads, seq_len_q, d_k])
            k (_type_): Key matrix (shape: [batch_size, num_heads, seq_len_k, d_k])
            v (_type_):  Value matrix (shape: [batch_size, num_heads, seq_len_v, d_v])
        """
        batch_size, n_head, seq_len, d_head = q.size()
        qk_t =  torch.matmul(q, k.transpose(-2,-1))
        scores = qk_t/torch.sqrt(torch.tensor(d_head, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention_weights = self.softmax(scores)
        
        output = torch.matmul(attention_weights, v)
        return output, attention_weights
        