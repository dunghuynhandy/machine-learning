import torch
from torch.utils.data import DataLoader, Dataset
import json
from PIL import Image
import os
from torchvision import transforms
class CLIPdataset(Dataset):
    def __init__(self, args):
        image_encoder_config = args.image_encoder["vit_Large"]
        image_size = image_encoder_config["image_size"]
        self.transform = transforms.Compose([
                                                transforms.Resize((image_size, image_size)),  # Resize images to a consistent size
                                                transforms.ToTensor(),          # Convert images to tensors
                                                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize images
                                            ])
        # with open("./annotations/captions_train2014.json") as f:
        #     self.captions = json.load(f)["annotations"]
        # with open("./annotations/captions_train2014.json") as f:
        #     images = json.load(f)["images"]
        # images = {item["id"]: item["file_name"] for item in images}
        # self.captions = [{**item, "image_path": os.path.join("/home/storage/datasets/COCO/train/", images[item["image_id"]])} for item in self.captions]
        # self.captions = [item for item in self.captions if os.path.exists(item["image_path"])]
        # with open("data.json", 'w') as f:
        #     json.dump(self.captions, f)
        with open("./data.json") as f:
            self.captions = json.load(f)
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        sample = self.captions[idx]
        sample_copy  = sample.copy()
        image = Image.open(sample_copy["image_path"])
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = self.transform(image)
        sample_copy["image"] = image
        return sample_copy

if __name__ == "__main__":
    dataset = CLIPdataset()
    print(dataset[1])