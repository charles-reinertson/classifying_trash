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
from torch.utils.data import Dataset, DataLoader
from utils import config
from ast import literal_eval

def create_csv():
    """
    create a csv labeled trash that contains the columns image filename with path, a partition (train, test, validate),
    a semantic label (ex. plastic cup), and a numeric label for the semantic label
    """
    anns_file_path = config('anns_file_path')
    csv_file = config('csv_file')

    # Read annotations
    with open(anns_file_path, 'r') as f:
        dataset = json.loads(f.read())

    categories = dataset['categories']
    anns = dataset['annotations']
    imgs = dataset['images']

    # name of csv file columns
    feature_list = ["filename", "partition", "semantic_label", "numeric_label", "segmentation", "area", "bbox"]

    # create a pandas dataframe with length number of images and width number of features in feature_list
    df = pd.DataFrame(None, index=np.arange(len(imgs)), columns=feature_list)

    # create a train, test, and validate split with the following listed weights
    df['partition'] = random.choices(
                population=['train', 'test', 'validate'],  # list to pick from
                weights=[0.6, 0.2, 0.2],  # weights of the population, in order
                k=len(imgs)  # amount of samples to draw
            )

    # go through each annotation and add label to dataframe
    for i, an in enumerate(anns):
        image = imgs[an['image_id']]
        category = categories[an['category_id']]
        
        # if we have not seem the same image before fill in filename, category, numeric_label, segmentation, area, and bbox of pandas dataframe
        if pd.isna(df.loc[image['id'], "filename"]):
            df.at[image['id'], 'filename'] = image['file_name']
            df.at[image['id'], 'semantic_label'] = [category['name']]
            df.at[image['id'], 'numeric_label'] = [category['id']]
            df.at[image['id'], 'segmentation'] = [an['segmentation']]
            df.at[image['id'], 'area'] = [an['area']]
            df.at[image['id'], 'bbox'] = [an['bbox']]
        # if we have seen the image add another category, numeric_label, segmentation, area, and bbox
        else:
            df.at[image['id'], 'semantic_label'].append(category['name'])
            df.at[image['id'], 'numeric_label'].append(category['id'])
            df.at[image['id'], 'segmentation'].append(an['segmentation'])
            df.at[image['id'], 'area'].append(an['area'])
            df.at[image['id'], 'bbox'].append(an['bbox'])

    df.to_csv(csv_file, index=False)

def get_train_val_test_loaders(num_classes):
    train, validate, test, _ = get_train_val_test_dataset(num_classes=num_classes)

    batch_size = config('cnn.batch_size')
    tr_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(validate, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return tr_loader, va_loader, te_loader, train.get_semantic_label

def get_train_val_test_dataset(num_classes):
    train = GarbageDataset('train', num_classes)
    validate = GarbageDataset('validate', num_classes)
    test = GarbageDataset('test', num_classes)

    # Resize
    train.X = resize(train.X)
    validate.X = resize(validate.X)
    test.X = resize(test.X)

    # Standardize
    standardizer = ImageStandardizer()
    standardizer.fit(train.X)
    train.X = standardizer.transform(train.X)
    validate.X = standardizer.transform(validate.X)
    test.X = standardizer.transform(test.X)

    # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
    train.X = train.X.transpose(0,3,1,2)
    validate.X = validate.X.transpose(0,3,1,2)
    test.X = test.X.transpose(0,3,1,2)

    return train, validate, test, standardizer

def get_unlabeled_loader(num_classes):
    unl = get_unlabeled_dataset(num_classes=num_classes)

    batch_size = config('cnn.batch_size')
    unl_loader = DataLoader(unl, batch_size=batch_size, shuffle=False)

    return unl_loader

def get_unlabeled_dataset(num_classes=5):
    train = GarbageDataset('train', num_classes)
    unl = GarbageDataset('unlabeled', num_classes)
    unl.X = resize(unl.X)

    standardizer = ImageStandardizer()
    standardizer.fit(train.X)
    unl.X = standardizer.transform(unl.X)
    unl.X = unl.X.transpose(0,3,1,2)

    return unl

def resize(X):
    """
    Resizes the data partition X to the size specified in the config file.
    Uses bicubic interpolation for resizing.

    Returns:
        the resized images as a numpy array.
    """
    image_dim = config('image_dim')
    # hard coded rgb image
    # Todo: change this to be more robust
    resized = np.ones((len(X), image_dim, image_dim, 3))
    #
    for i in range(len(X)):
        im = Image.fromarray(X[i])
        holder = im.resize(size=(image_dim, image_dim), resample=Image.BICUBIC)
        resized[i] = np.asarray(holder)
    return resized

class ImageStandardizer:
    """
    Channel-wise standardization for batch of images to mean 0 and variance 1.
    The mean and standard deviation parameters are computed in `fit(X)` and
    applied using `transform(X)`.

    X has shape (N, image_height, image_width, color_channel)
    """
    def __init__(self):
        super().__init__()
        self.image_mean = None
        self.image_std = None

    def fit(self, X):
        """
        Calculate the mean and standard deviation of each RGB layer. Save this
        to a text file
        """
        
        self.image_mean = np.mean(X, axis = (0, 1, 2))
        self.image_std = np.std(X, axis = (0, 1, 2))
        file1 = open("image_mean_std.txt","w")
        file1.write("Image mean: {}\n".format(self.image_mean))
        file1.write("Image std: {}".format(self.image_std))

    def fit_from_saved(self):
        file1 = open("image_mean_std.txt","r")
        text = file1.readlines()
        file1.close()
        # retrieve the saved mean from the training batch of images
        mean = text[0].split("[")[1].split("]")[0].split(" ")
        self.image_mean = [np.double(x) for x in mean]
        # retrieve the saved std from the training batch of images
        std = text[1].split("[")[1].split("]")[0][0:-1].split(" ")
        self.image_std = [np.double(x) for x in mean]

    def transform(self, X):
        """
        Normalize with Z score normalization
        """
        X = X - self.image_mean
        X = X / self.image_std
        
        # WAY TOO SLOW AND THE ABOVE DOES THE SAME THING
        # for i in range(X.shape[0]):
        #     for j in range(X.shape[1]):
        #         for k in range(X.shape[2]):
        #             for l in range(X.shape[3]):
        #                 X[i][j][k][l] -= self.image_mean[l]
        #                 if (self.image_std[l] != 0):
        #                      X[i][j][k][l] /= self.image_std[l]       

        return X

class GarbageDataset(Dataset):

    def __init__(self, partition, num_classes=10):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()

        if partition not in ['train', 'validate', 'test', 'unlabeled']:
            raise ValueError('Partition {} does not exist'.format(partition))

        np.random.seed(0)
        self.partition = partition
        self.num_classes = num_classes

        # Load in all the data we need from disk
        self.metadata = pd.read_csv(config('csv_file'), converters={"semantic_label": literal_eval,
                                                                    "numeric_label": literal_eval, 
                                                                    "segmentation": literal_eval,
                                                                    "area": literal_eval,
                                                                    "bbox": literal_eval
                                                                    })
        # convert columns 
        self.X, self.y = self._load_data()

        # self.semantic_labels = dict(zip(
        #     self.metadata['numeric_label'],
        #     self.metadata['semantic_label']
        # ))
        self.semantic_labels = self.__retrieve_labels__()

    def __retrieve_labels__(self):
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

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).long()

    def _load_data(self):
        """
        Loads a single data partition from file.
        """
        print("loading %s..." % self.partition)

        df = self.metadata[(self.metadata.partition == self.partition)]

        # TODO: uncomment the correct stuff

        X, y = [], []
        for i, row in df.iterrows():
            image = imread(os.path.join(config('image_path'), row['filename']))
            X.append(image)
            # y.append(row['numeric_label'])
            y.append(row['numeric_label'][0])

        # return np.array(X), np.array(y)
        return X, np.array(y)
        # return X, y

    def get_semantic_label(self, numeric_label):
        """
        Returns the string representation of the numeric class label (e.g.,
        the numberic label 1 maps to the semantic label 'sweaters').
        """
        return self.semantic_labels[numeric_label]

if __name__ == '__main__':
    ## Future note: check scipy imread and imresize
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    np.set_printoptions(precision=3)
    # edit this variabe to specify the number of classes in Taco dataset
    num_classes = 60
    # create a csv file that has all the necessary information for computer vision component
    create_csv()
    train, validate, test, standardizer = get_train_val_test_dataset(num_classes)
    print("Train:\t", len(train.X))
    print("Validate:\t", len(validate.X))
    print("Test:\t", len(test.X))
    print("Mean:", standardizer.image_mean)
    print("Std: ", standardizer.image_std)
