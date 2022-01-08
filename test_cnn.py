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
import imghdr

torch.manual_seed(36)
np.random.seed(36)
random.seed(36)

class Predict_image:
    def __init__(self, filename, model):
        self.filename = filename
        self.labels = retrieve_labels()
        self.img_loader = image_to_dataloader(retrieve_image(filename))
        self.img = Image.open(os.path.join(config('image_path'), filename))
        self.model = model

    def test_image(self):
        """
        run a single image through the model and return the output 
        """

        # run image through model and get output
        for X in self.img_loader:
            with torch.no_grad():
                output = self.model(X.float())
                predicted = predictions(output.data)
                print("Predicted: {} \n". format(self.labels[int(predicted)]))

        return self.labels[int(predicted)]
    
    def update_image(self, filename):
        self.filename = filename
        self.img_loader = image_to_dataloader(retrieve_image(filename))
        self.img = Image.open(os.path.join(config('image_path'), filename))

    def display_image(self, predicted=None):
        if predicted:
            # Call draw Method to add 2D graphics in an image
            I1 = ImageDraw.Draw(self.img)

            # Custom font style and font size
            myFont = ImageFont.truetype("/Library/Fonts/Arial.ttf", 150)
            
            # Add Text to an image
            I1.text((220, 500), predicted, font=myFont, fill=(20, 50, 50))
            
            # Display edited image
            self.img.show()
        else:
            self.img.show()



def load_model():
    """
    Return a mode restored to the latest checkpoint
    """
    model = CNN().float()

    # Attempts to restore the latest checkpoint
    print('Loading cnn...')
    return restore_latest_checkpoint(model, config('cnn.checkpoint'))

def continuous_image_input(model):
    while True:
        print("Enter file location of image:")
        print(">> ", end='')
        filename = str(input())

        try:
            img_type = imghdr.what(os.path.join(config('image_path'), filename))

            if img_type:
                break
            else:
                print("\nThis is not an image. Please try again")
        except:
            print("\nThis is an incorrect file path. Please try again")


    image_to_predict = Predict_image(filename, model)

    while True:
        predicted = image_to_predict.test_image()
        image_to_predict.display_image(predicted)

        while True:
            print("Enter another file location of image or 'exit' to quit program:")
            print(">> ", end='')
            filename = str(input())
            if filename == 'exit':
                return

            try:
                img_type = imghdr.what(os.path.join(config('image_path'), filename))

                if img_type:
                    break
                else:
                    print("\nThis is not an image. Please try again")
            except:
                print("\nThis is an incorrect file path. Please try again")

        
        image_to_predict.update_image(filename)



def main():
    # Data loaders
    # Model
    model = load_model()

    continuous_image_input(model)

    print("\nExiting program")

    


    


  

if __name__ == '__main__':
    main()
