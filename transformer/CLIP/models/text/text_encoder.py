import torch
from transformers import DistilBertTokenizer, DistilBertModel, AutoTokenizer

class DistilbBert(torch.nn.Module):
    def __init__(self, args):
        super(DistilbBert, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args["name"])
        self.model = DistilBertModel.from_pretrained(args["name"])
        self.fc = torch.nn.Linear(768, args["text_embedding_size"])
        
    def forward(self, x):
        x = self.tokenizer(x, return_tensors='pt', padding=True, truncation=True)
        x = x.to("cuda:0")
        with torch.no_grad():
            x = self.model(**x)
        x = x.last_hidden_state[:, 0, :]
        x = self.fc(x)
        return x
        
        
        
    