
import torch.nn.functional as F
from torch import nn
import torch
def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def contrastive_sigmoid_loss_function( text_projection,  image_projection):
    logits_per_text = (
            torch.matmul(text_projection, image_projection.t().to("cuda:0"))
        )
    logits_per_image = logits_per_text.t()
    eye = torch.eye(logits_per_text.size(0), device="cuda:0")
    m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
    loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
    nll = -torch.sum(loglik, dim=-1)
    loss = nll.mean()
    return loss