from torch import nn
from loguru import logger
from model.embedding import InputEmbedding, PositionEncoding
from model.layers import DecoderLayer
class Decoder(nn.Module):
    def __init__(self,
                 dec_vocal_size=30000,
                 d_model=512,
                 max_len=512,
                 n_head = 8,
                drop_prob = 0.5,
                dim_feedforward =1028,
                num_decoder_layers = 1
            ):
        super(Decoder, self).__init__()
        logger.debug("Debug Decoder")
        
        # input embedding
        self.input_embedding = InputEmbedding(dec_vocal_size, d_model)
        
        #https://towardsdatascience.com/understanding-positional-encoding-in-transformers-dc6bafc021ab
        self.position_encoding = PositionEncoding(d_model, max_len)
        
        #Encoder Block
        self.decoder_block =  nn.ModuleList([DecoderLayer(
                                d_model=d_model,
                                n_head = n_head,
                                dim_feedforward = dim_feedforward,
                                drop_prob = drop_prob
                                )for _ in range(num_decoder_layers)])
        self.lm_header = nn.Linear(d_model, dec_vocal_size)

    def forward(self, x, y, x_mask, y_mask):
        y = self.input_embedding(y)
        y = self.position_encoding(y)
        for decoder_layer in self.decoder_block:
            y = decoder_layer( x, y, x_mask, y_mask)
        y = self.lm_header(y)
        return y
        
        