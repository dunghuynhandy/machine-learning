import torch
from models.image.ViT import ViT
from models.text.text_encoder import DistilbBert
from loguru import logger
from models.projector import ProjectionHead
class CLIP(torch.nn.Module):
    def __init__(self, args):
        super(CLIP, self).__init__()
        self.dropout_prob = args.dropout_prob
        self.shared_dim = args.shared_dim
        self.image_encoder = ViT(args.image_encoder["vit_Large"]).to("cuda:0")
        self.text_encoder = DistilbBert(args.text_encoder["distilbert"]).to("cuda:0")
        
        self.image_projector = ProjectionHead(
            dropout=self.dropout_prob,
            projection_dim=self.shared_dim,
            embedding_dim=args.image_encoder["vit_Large"]['image_embedding_size']
            
        )
        
        self.text_projector = ProjectionHead(
            dropout=self.dropout_prob,
            projection_dim=self.shared_dim,
            embedding_dim=args.text_encoder["distilbert"]['text_embedding_size']
        )
        self.device = args.device
        
    def forward(self,v,t):
        v = self.image_encoder(v)
        logger.debug(f"image dim from Image encoder: {v.size()}")
        v = self.image_projector(v)
        logger.debug(f"image dim from Image projector: {v.size()}")
        
        t = self.text_encoder(t)
        logger.debug(f"text dim from Text encoder: {t.size()}")
        t = self.text_projector(t)
        logger.debug(f"text dim from Text projector: {t.size()}")
        
        return v, t
        
        
    