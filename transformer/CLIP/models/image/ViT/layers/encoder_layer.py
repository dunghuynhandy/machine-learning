import torch
from models.image.ViT.layers.multi_head_attention import MultiHeadAttention
class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_head, dropout_pro, dim_feedforward):
        super(EncoderLayer, self).__init__()
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout_pro)
        self.self_attention  = MultiHeadAttention(
                                                d_model = d_model,
                                                n_head=n_head
                                                )

        self.norm2 = torch.nn.LayerNorm(d_model)
        self.dropout2 = torch.nn.Dropout(dropout_pro)
        self.fc = torch.nn.Sequential(
                            torch.nn.Linear(d_model, dim_feedforward),
                            torch.nn.ReLU(),
                            torch.nn.Linear(dim_feedforward, d_model),
                        )
        
        
    def forward(self, x, mask=None):
        x_ = x
        x = self.norm1(x)
        x = self.self_attention(q=x, k=x, v=x, mask=mask)
        x = x + x_
        x = self.dropout1(x)
        
        x_ = x
        x = self.norm2(x)
        x = self.fc(x)
        x = self.dropout2(x)
        x = x + x_
        
        return x
        
    
    