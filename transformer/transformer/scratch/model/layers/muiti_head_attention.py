import torch
from torch import nn
from loguru import logger
from model.layers.scale_dot_product_attention import ScaledDotProductAttention
class MultiHeadAttention(torch.nn.Module):
    
    def __init__(self,
                 d_model,
                 n_head= 8,
                 ):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        #check model dimension must be divisible by number of hears
        assert d_model % n_head == 0, logger.error(f"d_model {d_model} must be divisible by number_headers {n_head}")
        self.scale_dot_product_attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        
        q = self.split_tensor(q)
        k = self.split_tensor(k)
        v = self.split_tensor(v)
        out, attention = self.scale_dot_product_attention(q=q, k=k, v=v, mask=mask)
        out = self.concat_tensor(out)
        out = self.linear(out)
        return out
    
    def split_tensor(self,tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()
        d_head = d_model//self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_head).transpose(1, 2)
        
        return tensor


    def concat_tensor(self,tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, n_head, length, d_head  = tensor.size()
        tensor = tensor.transpose(1, 2).contiguous()
            
        
        return tensor