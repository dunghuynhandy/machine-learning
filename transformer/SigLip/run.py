from data import CLIPdataset
from torch.utils.data import DataLoader, Dataset
from models import SigLip
import yaml
import argparse
from contrastive_sigmoid_loss import contrastive_sigmoid_loss_function
from loguru import logger
import torch
from tqdm import tqdm
logger.remove()
def run(args):
    dataset = CLIPdataset(args)
    dataloader = DataLoader(dataset, batch_size = args.bs)
    device = "cuda:0"
    args.device = device
    model = SigLip(args)
    optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=0.01
        )
    
    model = model.to(device)
    for  epoch in range(5):
        total_loss = 0
        for idx,batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            images = batch["image"]
            captions = batch["caption"]
            model.train()
            image_projections, text_projections = model(images, captions)
            loss = contrastive_sigmoid_loss_function(image_projections, text_projections)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if idx  % 10 == 0:
                print(f"{idx+1}|{len(dataloader)}: {loss.item()}")
        epoch_loss = total_loss/(idx+1)
        print(f"EPOCH {epoch} | LOSS {epoch_loss}")
            
        
        
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This is a sample Python script.")
    parser.add_argument("--bs", default=128,  help="Increase output verbosity")
    args = parser.parse_args()
    
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    args.image_encoder  = config["image_encoder"]
    args.text_encoder  = config["text_encoder"]
    args.dropout_prob = config["dropout_prob"]
    args.shared_dim = config["shared_dim"]
    run(args)
