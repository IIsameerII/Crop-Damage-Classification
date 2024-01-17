'''All the configrations of the project are stored here'''
# Import Nessecary Libraires
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import WeightedRandomSampler, Subset, random_split
from torch import nn
import os

# Setup hyperparameters
NUM_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_WORKERS = 0 #os.cpu_count()
TRAIN_SIZE = 0.8
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# HIDDEN_LAYERS = 10 This is not required for this project
MODEL_NAME = 'EfficientNetB0'
EARLY_STOP = 10
# Get a set of pretrained model weights
weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # "DEFAULT" = best available weights
model = torchvision.models.efficientnet_b0(weights=weights).to(DEVICE)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# Rewrite the dataset variable to transfrom using auto transform 
dataset = ImageFolder(root=r'images\images\train',
                            transform=None,
                            target_transform=None)

class_names = dataset.classes

# Setup directories
train_dir = r"C:\Users\SameerAhamed\Downloads\cgiar-crop-damage-classification-challenge\images\images\train"
validation_dir = r"C:\Users\SameerAhamed\Downloads\cgiar-crop-damage-classification-challenge\images\images\valid"
MODEL_SAVE_PATH = r'C:\Users\SameerAhamed\Downloads\cgiar-crop-damage-classification-challenge\models'

'''Change this for each and every model'''
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2, inplace=True),
    nn.Linear(in_features=1280, # feature vector coming in EfficientNetB0 and EfficientNetB1
            #   in_features=1408, # Feature vector for EfficientNetB2
              out_features=len(class_names))).to(DEVICE) # how many classes do we have?

# Get the transforms used to create our pretrained weights
auto_transforms = weights.transforms()

# Define the mean and std
mean = torch.tensor([0.485, 0.456, 0.406])  # mean values
std = torch.tensor([0.229, 0.224, 0.225])   # std values

# Change the transforms here
manual_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),    # Randomly flips the image horizontally
    transforms.RandomRotation(30),        # Randomly rotates the image by up to 30 degrees
    transforms.RandomGrayscale(p=0.2),    # Randomly converts images to grayscale with a probability of 0.2
    transforms.RandomAdjustSharpness(sharpness_factor=2,p=0.5),
    transforms.RandomAffine(              # Applies random affine transformations
        degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10
    ),
    transforms.RandomPerspective(         # Applies a random perspective transformation
        distortion_scale=0.5, p=0.5
    ),
    transforms.RandomApply([transforms.ColorJitter(               # Randomly varies brightness, contrast, saturation, and hue
        brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
    ),
    transforms.GaussianBlur(              # Applies a Gaussian blur with a random kernel size
        kernel_size=(5, 9), sigma=(0.1, 5)
    )],p=0.25),
    transforms.Resize([256,256],interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean,std)
])

# Split the dataset into training and validation sets
train_size = int(TRAIN_SIZE * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Custom Subset class to apply different transformations
class TransformedSubset(Subset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

# Apply transformations
train_dataset = TransformedSubset(train_dataset, transform=manual_transform)
test_dataset = TransformedSubset(test_dataset, transform=auto_transforms)
transform_used = train_dataset.transform # This command is to log the transform that is used in MLFLow
'''
Sampler Code Starts
'''
# Calculate weights for each class
class_counts = torch.bincount(torch.tensor([sample[1] for sample in train_dataset]))
class_weights = 1. / class_counts.float()
weightss = class_weights[torch.tensor([sample[1] for sample in train_dataset])]

sampler = WeightedRandomSampler(weights=weightss,
                                num_samples=len(weightss),
                                replacement=True)
# sampler = None # Comment if you are not using sampler
'''
Sampler Code Ends
'''