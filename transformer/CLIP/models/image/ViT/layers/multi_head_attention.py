import torch


class MultiHeadAttention(torch.nn.Module):
    
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model%n_head ==0, print("dimension must be dividable by n_head")
        self.d_model = d_model
        self.d_head = d_model//n_head
        self.n_head = n_head
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.fc_out = torch.nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, q, k, v, mask):
        q_k_t = torch.matmul(q, k.transpose(-2,-1))
        scores = q_k_t/torch.sqrt(torch.tensor(self.d_head, dtype = torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        return output, attention_weights
        
    def forward(self, q, k, v, mask=None):
        
        batch_size, no_token, d_model = q.size()
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
       
        q = q.view(batch_size, no_token, self.n_head, self.d_head).transpose(1,2)
        k = k.view(batch_size, no_token, self.n_head, self.d_head).transpose(1,2)
        v = v.view(batch_size, no_token, self.n_head, self.d_head).transpose(1,2)
        
        output, attention_weights = self.scaled_dot_product_attention(q=q, k=k, v=v, mask=mask)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, no_token, d_model)
        output = self.fc_out(output)
        return output
    