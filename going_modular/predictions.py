"""
Utility functions to make predictions.
"""
import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import config
from typing import List, Tuple
from PIL import Image

# Set device
device = config.DEVICE

# Predicition
import pandas as pd
from PIL import Image

# Input filepaths here
valid_dir = config.validation_dir
valid_csv = pd.read_csv(r'Test.csv')
pred_df = pd.DataFrame({'ID':[],
                        'DR':[],
                        'G':[],
                        'ND':[],
                        'WD':[],
                        'other':[]})

# Iterate in each image and get the prediction
config.model.eval()
with torch.inference_mode():
    for i,file_name in enumerate(valid_csv['filename']):
        # Store the variable file_row
        ID = valid_csv.iloc[i,0]
        # print(ID)

        # Concatenate to get the file_path
        file_path = valid_dir + file_name
        # print(i,filepath,valid_csv.iloc[[i]])
        
        # Open image
        img = Image.open(file_path)
        
        # Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
        transformed_image = config.auto_transforms(img).unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = config.model(transformed_image.to(device))

        # Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
        target_image_pred_probs = list(torch.softmax(target_image_pred, dim=1).to('cpu').numpy())

        # print(target_image_pred_probs[0][0])

        # New row to be added
        new_row = {'ID': ID,
                    'DR':target_image_pred_probs[0][0],
                    'G':target_image_pred_probs[0][1],
                    'ND':target_image_pred_probs[0][2],
                    'WD':target_image_pred_probs[0][3],
                    'other':target_image_pred_probs[0][4]}
        
        # break
        # pred_df = pred_df.append(new_row,ignore_index = True)
        pred_df = pd.concat([pred_df,pd.DataFrame([new_row])],ignore_index=True)
pred_df