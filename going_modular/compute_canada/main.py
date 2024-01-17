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
NUM_WORKERS = 0 # os.cpu_count()
print(NUM_WORKERS)
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

# Setup directories
train_dir = r"/home/asameer/projects/def-hamilton/asameer/cgiar-crop-damage-classification-challenge/images/images/train"
validation_dir = r"/home/asameer/projects/def-hamilton/asameer/cgiar-crop-damage-classification-challenge/images/image/valid"
MODEL_SAVE_PATH = r'/home/asameer/projects/def-hamilton/asameer/cgiar-crop-damage-classification-challenge/models'

'''COMMENT THIS WHEN RUNNING IN COMPUTE CANADA'''
# Setup directories
# train_dir = r"C:\Users\SameerAhamed\Downloads\cgiar-crop-damage-classification-challenge\images\images\train"
# validation_dir = r"C:\Users\SameerAhamed\Downloads\cgiar-crop-damage-classification-challenge\images\images\valid"
# MODEL_SAVE_PATH = r'C:\Users\SameerAhamed\Downloads\cgiar-crop-damage-classification-challenge\models'

# Rewrite the dataset variable to transfrom using auto transform 
dataset = ImageFolder(root=train_dir,
                            transform=None,
                            target_transform=None)

class_names = dataset.classes


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
###########################################################################

"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(  # the dataset comes in like that
    train_dataset,
    test_dataset, 
    batch_size: int, 
    num_workers: int,
    sampler=None
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """

  # Get class names
  # class_names = train_dataset.classes

  # Turn images into data loaders, use sampler if not none
  if sampler != None:
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
  else:
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

  test_dataloader = DataLoader(
      test_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader

######################################################################################

"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import torchvision


from tqdm.auto import tqdm
from typing import Dict, List, Tuple

import mlflow
import mlflow.pytorch

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               scheduler, # Added scheduler
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(tqdm(dataloader,desc='Model Training loop')):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              scheduler: torch.optim.lr_scheduler) -> Tuple[float, float]:
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(tqdm(dataloader,desc='Model Testing loop')):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    scheduler.step(test_loss)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          transform_used: torchvision.transforms,
          seed_number,
          exp_num:int,
          model_name:str,
          model_save_path:str,
          early_stop:int,
          scheduler = None,
          gamma = None) -> Dict[str, List]:
    
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    # run_name = 'Exp_'+str(exp_num)
    
    with mlflow.start_run():
      try:
        # mlflow.set_tag("mlflow.runName", run_name)
        # start_time = timer()
        # Create empty results dictionary
        results = {"train_loss": [],
                    "train_acc": [],
                    "test_loss": [],
                    "test_acc": []
        }
        
        # Make sure model on target device
        model.to(device)

        # Get the learning rate of the model
        for param_group in optimizer.param_groups: lr = param_group['lr']
        # Get the batch size of the model which is to be trained
        BATCH_SIZE = train_dataloader.batch_size
        # Get the optimizer that we are using for this experiment
        optimizer_name = optimizer.__class__.__name__
        
        mlflow.log_param('learning_rate', lr)
        mlflow.log_param('batch_size', BATCH_SIZE)
        mlflow.log_param('epochs',epochs)
        mlflow.log_param('optimizer',optimizer_name)

        if scheduler != None:
          scheduler_name = scheduler.__class__.__name__
          mlflow.log_param('factor',scheduler.factor)
          mlflow.log_param('patience',scheduler.patience)
          mlflow.log_param('scheduler',scheduler_name)


        mlflow.log_param('device',device)
        mlflow.log_param('transforms',transform_used)
        mlflow.log_param('seed number',seed_number)
        mlflow.log_param('Model Name',model_name)
        mlflow.log_param('experiment number',int(exp_num))
        mlflow.log_param('early stop number',early_stop)
        # mlflow.pytorch.log_model(model, "model")
        
        # Intializing the best test loss
        best_test_loss = 1000
        stop_count = 0
        mlflow.log_metric('early stop',False)

        # Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs),desc='Total Training'):
            train_loss, train_acc = train_step(model=model,
                                              dataloader=train_dataloader,
                                              loss_fn=loss_fn,
                                              optimizer=optimizer,
                                              scheduler=scheduler,
                                              device=device)
            # Scheduler code
            if scheduler != None:
              last_lr = optimizer.param_groups[0]['lr']
              mlflow.log_metric('lr_scheduler',last_lr,step=epoch) # If planning to remove the lr 
            
            # Testing code
            test_loss, test_acc = test_step(model=model,
              dataloader=test_dataloader,
              loss_fn=loss_fn,
              device=device,scheduler=scheduler)
            
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('test_loss', test_loss, step=epoch)

            # Saving the best model
            if test_loss <= best_test_loss:
                print(f'epoch # {epoch} is Best Epoch yet! ')
                torch.save(model, model_save_path+'best_'+str(exp_num)+'.pt')
                best_test_loss = test_loss
                best_train_loss = train_loss
                best_epoch = epoch
                mlflow.log_metric('best test loss',best_test_loss)
                mlflow.log_metric('best train loss',best_train_loss)
                mlflow.log_metric('best epoch', best_epoch)
                stop_count = 0
            else:
                stop_count += 1
            

            # Print out what's happening
            print(
              f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"test_loss: {test_loss:.4f} | "
              f"test_acc: {test_acc:.4f}"
            )

            # Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)
            # Change this and put this down
            if stop_count==early_stop-1:
                mlflow.log_metric('early stop',True)
                print('Early Stop')
                return results

        return results
      except KeyboardInterrupt:
          mlflow.log_metric('early stop',True)
          print('Training Stopped')
          return results
      
######################################################################################
      
import os

def count_files_in_directory(directory):
    """
    Count the number of files in the given directory.

    :param directory: Path to the directory
    :return: Number of files in the directory
    """
    if not os.path.isdir(directory):
        print(f"The specified path {directory} is not a directory.")
        return

    # Count the number of files
    file_count = len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])
    return file_count

###########################################################################################
"""
Trains a PyTorch image classification model using device-agnostic code.
"""
print('Script Started')
import os
import torch
import data_setup, engine, utils
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torchvision import transforms
from count_files_in_directory import count_files_in_directory
print('Libraires imported')

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    sampler=sampler
)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Setup training and save the results
results = engine.train(model=model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer, # You can put an optimizer and scheduler here
                       scheduler=scheduler,
                       gamma=None,
                       loss_fn=loss_fn,
                       epochs=NUM_EPOCHS,
                       device=DEVICE,
                       transform_used=transform_used,
                       seed_number=SEED,
                       exp_num=count_files_in_directory(MODEL_SAVE_PATH),
                       model_name = MODEL_NAME,
                       model_save_path = MODEL_SAVE_PATH,
                       early_stop=EARLY_STOP,
                       )

# Predict output and save in .csv file

