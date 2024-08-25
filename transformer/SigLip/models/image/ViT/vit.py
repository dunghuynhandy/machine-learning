import torch
from loguru import logger
from models.image.ViT.layers import PositionEncoding
from models.image.ViT.layers import EncoderLayer

class ViT(torch.nn.Module):
    def __init__(self, args):
        super(ViT, self).__init__()
        self.patch_size = args["patch_size"]
        self.image_size = args["image_size"]
        self.d_model = args["hidden_size"]
        self.n_head = args["n_head"]
        self.dropout_pro = args["dropout_pro"]
        self.dim_feedforward = args["dim_feedforward"]
        self.image_embedding_size = args["image_embedding_size"]
        assert self.image_size % self.patch_size  == 0, "Image dimensions must be divisible by the patch size."
        num_patches = (self.image_size // self.patch_size) ** 2
        self.patch_dim = 3 * self.patch_size ** 2
        self.linear_projection = torch.nn.Linear(self.patch_dim,self.d_model)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.position_encoding  = PositionEncoding(num_patches+1, self.d_model)
        
        self.transformer_encoder_block = torch.nn.ModuleList([
            EncoderLayer(d_model = self.d_model,
                         n_head=self.n_head,
                         dropout_pro= self.dropout_pro,
                         dim_feedforward = self.dim_feedforward) 
            for _ in range(args['num_layers'])
            ])
        self.dropout = torch.nn.Dropout(self.dropout_pro)
        self.fc_head = torch.nn.Linear(self.d_model, self.image_embedding_size)

        
    
    def forward(self, images):
        logger.debug(f"original image size [b, c, h, w]: {images.size()}")
        images = self.get_batch_patches(images)
        logger.debug(f"[b, n_patch, c, h, w]: {images.size()}")
        batch_size, no_patches, c, h, w =images.size()
        images = images.reshape(batch_size, no_patches,c*h*w)
        logger.debug(f"[b, n_patch, c*h*w]: {images.size()}")
        images = images.to("cuda:0")
        images = self.linear_projection(images)
        logger.debug(f"[b, n_patch, d_model]: {images.size()}")
        logger.debug(f"cls token dim [1, 1, d_model]: {self.cls_token.size()}")
        batch_cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        logger.debug(f"batch_cls_tokens dim [b, 1, d_model]: {batch_cls_tokens.size()}")
        images = torch.concat([batch_cls_tokens, images], dim =1 )
        logger.debug(f"batch_cls_tokens + images dim: [b, n_patch+1, c*h*w]: {images.size()}")
        images = self.position_encoding(images)
        
        for transformer_enc_layer in self.transformer_encoder_block:
            images = transformer_enc_layer(images)
        cls_token = images[:, 0]
        output = self.fc_head(cls_token)
        return output
    
    
    
    def create_image_patches(self, image):
        batch_size, channels, height, width = image.size()
        patches = []
        for h in range(0, height, self.patch_size):
            for w in range(0, width, self.patch_size):
                patch = image[:, :, h:h+self.patch_size, w:w+self.patch_size]
                patches.append(patch)
        patches = torch.cat(patches, dim=0)
        return patches
    
    def get_batch_patches(self, input_batch_image):
        batch_patches = []
        for i in range(input_batch_image.shape[0]):
            input_image = input_batch_image[i].unsqueeze(0)
            patches = self.create_image_patches(input_image)
            batch_patches.append(patches)
        return torch.stack(batch_patches, dim=0).float()
    
    