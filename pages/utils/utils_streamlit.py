import torch
from torchvision import datasets

def load_model():
    model = torch.load(r'models\best_5.pt')
    return model

def create_dataset_valid(path):
  valid_dataset = datasets.ImageFolder(path)
  # Not transfroming here. It will be transformed when the image is fed
  # to the prediciton function (single image). This was done to maintain consistency

  return valid_dataset