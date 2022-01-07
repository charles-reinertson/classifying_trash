"""
Charles Reinertson
Better Waste Project
Taco Dataset
    Class wrapper for interfacing with the dataset of garbage images
"""
import os
import json
import numpy as np
import pandas as pd
import torch
import random
from imageio import imread
from PIL import Image
from utils import config
from ast import literal_eval
from torch.utils.data import DataLoader
from dataset import resize, ImageStandardizer

def retrieve_labels():
        """
        Return a dictionary with keys as numeric labels and values as semantic labels
        """
        anns_file_path = config('anns_file_path')
        
        # Read annotations
        with open(anns_file_path, 'r') as f:
            dataset = json.loads(f.read())

        categories = dataset['categories']

        labels = {}
        for value in categories:
            labels[value['id']] = value['name']

        return labels

def retrieve_image(filename):
    """
    Retrieve the image at filename after being properly transformed
    """
    image = [imread(os.path.join(config('image_path'), filename))]

    # Resize
    image = resize(image)

    # Standardize
    standardizer = ImageStandardizer()
    standardizer.fit_from_saved()
    image = standardizer.transform(image)

    # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
    image = image.transpose(0,3,1,2)

    return image

def image_to_dataloader(image):
    """
    Return a dataloader with the single "image".
    """
    img_loader = DataLoader(image, batch_size=1)

    return img_loader


