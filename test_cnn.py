"""
EECS 445 - Introduction to Machine Learning
Fall 2020 - Project 2
Train CNN
    Trains a convolutional neural network to classify images
    Periodically outputs training information, and saves model checkpoints
    Usage: python train_cnn.py
"""
import torch
import numpy as np
import random
from prepare_one_image import retrieve_labels, retrieve_image, image_to_dataloader
from model.cnn import CNN
from train_common import *
from utils import config
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

torch.manual_seed(36)
np.random.seed(36)
random.seed(36)

def test_image(image, model, actual):
    """
    run a single image through the model and return the output and whether it was correct
    or not.
    """

    # run image through model and get output
    output = model(image)

    return output, (output == actual)




def main():
    # Data loaders
    print("Enter file location of image:")
    print(">> ", end='')
    # filename = str(input())
    filename = "batch_1/headshot.JPG"
    
    image = retrieve_image(filename)
    img_loader = image_to_dataloader(image)
    labels = retrieve_labels()

    # Model
    model = CNN().float()

    # Attempts to restore the latest checkpoint
    print('Loading cnn...')
    model = restore_latest_checkpoint(model, config('cnn.checkpoint'))

    for X in img_loader:
        with torch.no_grad():
            output = model(X.float())
            predicted = predictions(output.data)
            print("Predicted: {}". format(labels[int(predicted)]))


    img = Image.open(os.path.join(config('image_path'), filename))
    # Call draw Method to add 2D graphics in an image
    I1 = ImageDraw.Draw(img)

    # Custom font style and font size
    myFont = ImageFont.truetype("/Library/Fonts/Arial.ttf", 150)
    
    # Add Text to an image
    I1.text((220, 500), labels[int(predicted)], font=myFont, fill=(20, 50, 50))
    
    # Display edited image
    img.show()


  

if __name__ == '__main__':
    main()
