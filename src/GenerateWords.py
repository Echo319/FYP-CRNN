import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random # to create random data
from PIL import Image # saving raw images 

# Change NUM to get a number of words.
# TODO: Drop the images folder before running
# TODO: Argument for filter
# TODO: print letters onto a canvas instead of resize

def read_prep_train_data(dataset_name):
    read = pd.read_csv('../Dataset/' + dataset_name, header=None)
    #split data and labels
    train_data = read.iloc[:, 1:]
    train_labels = read.iloc[:, 0]
    #reshape labels 
    #train_labels = pd.get_dummies(train_labels)
    # get as numpy arrays
    train_data = train_data.values
    train_labels = train_labels.values 
    # apply rotation to dataset
    train_data = np.apply_along_axis(rotate, 1, train_data)
    #return tuple
    return train_data, train_labels

def rotate(image):
    image = image.reshape([28, 28])
    image = np.fliplr(image)
    image = np.rot90(image)
    return image.reshape([28 * 28])

def map_labels(map_loc, train_labels):
    # get map values the only way i know
    with open(map_loc) as inputfile:
        label_ascii = [line.strip().split() for line in inputfile.readlines()]
    # get as dictionary
    x= {}
    for i in label_ascii:
        x.update({i[0]:i[1]})
    # get labels as string then convert to ascii characters
    train_labels = train_labels.astype(str)
    for i in range(len(train_labels)):
        train_labels[i] = chr(int(x.get(train_labels[i])))
    return train_labels


print("Reading Data")
# get letters 
images, labels = read_prep_train_data('emnist-byclass-train.csv')
# change lables to letters 
labels = map_labels('../Dataset/emnist-byclass-mapping.txt', labels)
print("Data read and lables mapped")

print("Reading common words")
# get common words as list
words = [line.rstrip('\n') for line in open('../dataset/common_words.txt')]
print("Got common words")
words = list(filter(lambda x: len(list(x)) <= 8 and len(list(x)) >= 3, words))
print("Stripped words less than 3 chars and more than 8")

csv_imgs = []
# Test
NUM = 10
print("Generating {} words".format(NUM))
for word in words:
    # Always lowercase to increase performance
    word = word.lower()

    # Get a collection of letters matching the word
    letters = list(word)
    letter_indicies=[]
    for i in letters :
            indicies = np.where(labels == i)[0]
            letter_indicies.append(indicies)

    # make images using the random images
    for i in range(NUM):
        rand_letters = []
        # get the letters 
        for indicies in letter_indicies :
            # incase special characters are included in words
            if(len(indicies) > 0):
                rand_letters.append((images[random.choice(indicies)].reshape([28,28])))
        # put them as 1 image 
        word_as_array = np.concatenate(rand_letters ,axis=1)
        im = Image.fromarray(np.uint8(word_as_array))
        im = im.resize([len(word) * 28 ,64])
        new_im = Image.new('L', (252, 64), 0)
        new_im.paste(im, (0,0))
        im = new_im
        
        PATH = "../Dataset/Words/train/{}-{}.png".format(word, i)
        im.save(PATH)

# Train reccomended 20% 
NUM = 2   
for word in words:
    # Always lowercase to increase performance
    word = word.lower()
    # Get a collection of letters matching the word
    letters = list(word)
    letter_indicies=[]
    for i in letters :
            indicies = np.where(labels == i)[0]
            letter_indicies.append(indicies)

    # make images using the random images
    for i in range(NUM):
        rand_letters = []
        # get the letters 
        for indicies in letter_indicies :
            # incase special characters are included in words
            if(len(indicies) > 0):
                rand_letters.append((images[random.choice(indicies)].reshape([28,28])))
        # put them as 1 image 
        word_as_array = np.concatenate(rand_letters ,axis=1)
        im = Image.fromarray(np.uint8(word_as_array))
        im = im.resize([len(word) * 28 ,64])
        new_im = Image.new('L', (252, 64), 0)
        new_im.paste(im, (0,0))
        im = new_im
        
        PATH = "../Dataset/Words/test/{}-{}.png".format(word, i)
        im.save(PATH)
        
# predict test? 
NUM = 1
for word in words:
    # Always lowercase to increase performance
    word = word.lower()

    # Get a collection of letters matching the word
    letters = list(word)
    letter_indicies=[]
    for i in letters :
            indicies = np.where(labels == i)[0]
            letter_indicies.append(indicies)

    # make images using the random images
    for i in range(NUM):
        rand_letters = []
        # get the letters 
        for indicies in letter_indicies :
            # incase special characters are included in words
            if(len(indicies) > 0):
                rand_letters.append((images[random.choice(indicies)].reshape([28,28])))
        # put them as 1 image 
        word_as_array = np.concatenate(rand_letters ,axis=1)
        im = Image.fromarray(np.uint8(word_as_array))
        im = im.resize([len(word) * 28 ,64])
        new_im = Image.new('L', (252, 64), 0)
        new_im.paste(im, (0,0))
        im = new_im
        
        PATH = "../Dataset/Words/predict/{}-{}.png".format(word, i)
        im.save(PATH)
print("All the words saved")
