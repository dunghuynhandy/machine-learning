
import torch.nn.functional as F
from torch import nn
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def contrastive_clip_loss_function( text_projection,  image_projection, temperature_value):
    logits = (text_projection @ image_projection.T) / temperature_value
    images_similarity = image_projection @ image_projection.T
    texts_similarity = text_projection @ text_projection.T
    targets = F.softmax( (images_similarity + texts_similarity) / 2 * temperature_value, dim=-1 )
    texts_loss = cross_entropy(logits, targets, reduction='none')
    images_loss = cross_entropy(logits.T, targets.T, reduction='none')
    loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
    return loss.mean()