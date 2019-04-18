import cv2
import itertools, os, time
import numpy as np
from Model import get_Model
from Parameter import letters, img_h, img_w
import argparse
from keras import backend as K
K.set_learning_phase(0)

def decode_label(out):
    out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    outstr = ''
    for i in out_best:
        if i < len(letters):
            outstr += letters[i]
    return outstr

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight", help="weight file directory",
                    type=str, default="../final_weights.hdf5/")
parser.add_argument("-t", "--test_img", help="Test image directory",
                    type=str, default="../Dataset/words/predict/")
                    # "./Data/images/predict/"
args = parser.parse_args()

# Get CRNN model
model = get_Model(training=False)

# Load weights
try:
    model.load_weights(args.weight)
    print("...Previous weight data...")
except:
    raise Exception("No weight file!")

# Get predict/test set
test_dir = args.test_img
test_imgs = os.listdir(args.test_img)
# Metrics
total = 0
acc = 0
letter_total = 0
letter_acc = 0
# Timer
start = time.time()

for test_img in test_imgs:
    img = cv2.imread(test_dir + test_img, 0)
    # Transform identical to generator
    # img = cv2.resize(img, (img_w, 28))
    img = cv2.resize(img, (img_w, img_h))
    img_pred = img.astype(np.float32)
    img_pred = (img_pred / 255.0) * 2.0 - 1.0
    img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred, axis=-1)
    img_pred = np.expand_dims(img_pred, axis=0)
    net_out_value = model.predict(img_pred)
    pred_texts = decode_label(net_out_value)

    for i in range(min(len(pred_texts), len(test_img[0:-6]))):
        if pred_texts[i] == test_img[i]:
            letter_acc += 1
    letter_total += max(len(pred_texts), len(test_img[0:-6]))

    if pred_texts == test_img[0:-6]:
        acc += 1
    total += 1
    print('Predicted: %s  /  True: %s' % (pred_texts, test_img[0:-6]))
    
end = time.time()
total_time = (end - start)
print("Time : ",total_time / total)
print("ACC : ", acc / total)
print("letter ACC : ", letter_acc / letter_total)