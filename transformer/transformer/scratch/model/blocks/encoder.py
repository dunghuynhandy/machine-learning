from torch import nn
from loguru import logger
from model.embedding import InputEmbedding, PositionEncoding
from model.layers import EncoderLayer
class Encoder(nn.Module):
    def __init__(self,
                 enc_voc_size=30000,
                 d_model=512,
                 max_len=512,
                 n_head = 8,
                drop_prob = 0.5,
                dim_feedforward =1028,
                num_encoder_layers = 1
            ):
        super(Encoder, self).__init__()
        logger.debug("Debug Encoder")
        
        # input embedding
        self.input_embedding = InputEmbedding(enc_voc_size, d_model)
        self.position_encoding = PositionEncoding(d_model, max_len)
        
        #Encoder Block
        self.encoder_block =  nn.ModuleList([EncoderLayer(
                                d_model=d_model,
                                n_head = n_head,
                                dim_feedforward = dim_feedforward,
                                drop_prob = drop_prob
                                )for _ in range(num_encoder_layers)])

    def forward(self, x, mask):
        logger.debug(f"input size: {x.size()}")
        #embedding and position encoding
        x = self.input_embedding(x)
        x = self.position_encoding(x)
        for enc_layer in self.encoder_block:
            x = enc_layer(x, mask)
        return x
        