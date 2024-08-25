from torch import nn
from loguru import logger
import torch
from model.blocks import Encoder, Decoder
class Transformer(nn.Module):
    def __init__(self,
        enc_voc_size = 30000,
        dec_vocal_size = 30000,
        d_model=512,
        src_padding_token = 1,
        tgt_padding_token = 1,
        dim_feedforward = 1024,
        drop_prob =0.2,
        num_encoder_layers = 6,
        num_decoder_layers = 6,
        n_head = 8,
        max_len = 512
        ):
        super(Transformer, self).__init__()
        logger.debug("setting model")
        self.TransformerEncoder =Encoder(
                enc_voc_size=enc_voc_size,
                d_model = d_model,
                dim_feedforward = dim_feedforward,
                drop_prob = drop_prob,
                num_encoder_layers = num_encoder_layers,
                n_head = n_head,
                max_len = max_len
                )

        self.TransformerDecoder =Decoder(
                enc_voc_size=dec_vocal_size,
                d_model = d_model,
                dim_feedforward = dim_feedforward,
                drop_prob = drop_prob,
                num_decoder_layers = num_decoder_layers,
                n_head = n_head,
                max_len = max_len
                )

        self.src_padding_token = src_padding_token
        self.tgt_padding_token = tgt_padding_token
        
    def forward(self, x, y):
        logger.debug("x")
        logger.debug(x)
        x_mask = self.make_x_mask(x)
        logger.debug("x_mask")
        logger.debug(x_mask)
        x = self.TransformerEncoder(x, x_mask)
        logger.debug("x_enc")
        logger.debug(x.size())
        y_mask = self.make_y_mask(y)
        logger.debug("y")
        logger.debug(y)
        logger.debug("y_mask")
        logger.debug(y_mask)
        y = self.TransformerDecoder(x, y, x_mask, y_mask)
        return y
        
        
        
    
    def make_x_mask(self, x):
        x_mask = (x != self.src_padding_token).unsqueeze(1).unsqueeze(2)
        return x_mask
    
    def make_y_mask(self, y):
        y_padding_mask = (y != self.tgt_padding_token).unsqueeze(1).unsqueeze(2)
        device =  y_padding_mask.device
        y_len = y.shape[1]
        
        y_sub_mask = torch.tril(torch.ones(y_len, y_len)).type(torch.ByteTensor)
        y_sub_mask = y_sub_mask.to(device)
        y_mask = y_padding_mask & y_sub_mask
        return y_mask
    