import cv2
import os, random
import numpy as np
from Parameter import letters
from Parameter import LOWER_CASE

# # Input data generator
def labels_to_text(labels):    
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

# the name of the image to list of char
# Translation of characters to unique integer values
def text_to_labels(text):
    ret = []
    for char in text:
        ret.append(LOWER_CASE.find(char))
    return ret


class DataGenerator:
    def __init__(self, img_dirpath, img_w, img_h,
                 batch_size, downsample_factor, max_text_len=9):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath                  # image dir path
        self.img_dir = os.listdir(self.img_dirpath)     # images list
        self.n = len(self.img_dir)                      # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w))
        self.texts = []

    def build_data(self):
        print(self.n, " Image Loading start...")
        for i, img_file in enumerate(self.img_dir):
            # read image in grayscale
            img = cv2.imread(self.img_dirpath + img_file, 0)
            # resize the imaget to 128x64
            img = cv2.resize(img, (self.img_w, self.img_h))
            # return the image as numpyarray
            img = img.astype(np.float32)
            # normalize
            img = (img / 255.0) * 2.0 - 1.0
            # Add image to collections
            self.imgs[i, :, :] = img
            # Add label to collection (Strip of file extension and "-0" combo)
            self.texts.append(img_file[0:-6])
        print(len(self.texts) == self.n)
        print(self.n, " Image Loading finish...")

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):      
        while True:
            # Want everything to be this shape as this is the important bit
            X_data = np.ones([self.batch_size, self.img_w, self.img_h, 1])     # (bs, 128, 64, 1)
            Y_data = np.ones([self.batch_size, self.max_text_len])             # (bs, 9)
            input_length = np.ones((self.batch_size, 1)) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1))           # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()
                img = img.T
                img = np.expand_dims(img, -1)
                X_data[i] = img
                Y_data[i, 0:len(text)] = text_to_labels(text)
                label_length[i] = len(text)

            inputs = {
                'the_input': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1)
                'label_length': label_length  # (bs, 1)
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1)
            yield (inputs, outputs)