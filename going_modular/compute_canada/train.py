"""
Trains a PyTorch image classification model using device-agnostic code.
"""
print('Script Started')
import os
import torch
import data_setup, engine, utils
import config
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torchvision import transforms
from count_files_in_directory import count_files_in_directory
print('Libraires imported')
# Get the model from configs
model = config.model

class_names = config.class_names

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_dataset=config.train_dataset,
    test_dataset=config.test_dataset,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    sampler=config.sampler
)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Setup training and save the results
results = engine.train(model=config.model,
                       train_dataloader=train_dataloader,
                       test_dataloader=test_dataloader,
                       optimizer=optimizer, # You can put an optimizer and scheduler here
                       scheduler=scheduler,
                       gamma=None,
                       loss_fn=loss_fn,
                       epochs=config.NUM_EPOCHS,
                       device=config.DEVICE,
                       transform_used=config.transform_used,
                       seed_number=config.SEED,
                       exp_num=count_files_in_directory(config.MODEL_SAVE_PATH),
                       model_name = config.MODEL_NAME,
                       model_save_path = config.MODEL_SAVE_PATH,
                       early_stop=config.EARLY_STOP,
                       )

# Predict output and save in .csv file
