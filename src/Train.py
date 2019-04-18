from keras import backend as K
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from DataGenerator import DataGenerator
from Model import get_Model
from Parameter import *
import time
K.set_learning_phase(0)

# # Model description and training

model = get_Model(training=True)

train_file_path = '../Dataset/Words/train/'
x_train = DataGenerator(train_file_path, img_w, img_h, batch_size, downsample_factor)
x_train.build_data()

valid_file_path = '../Dataset/Words/test/'
x_val = DataGenerator(valid_file_path, img_w, img_h, val_batch_size, downsample_factor)
x_val.build_data()

ada = Adadelta()

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
# save @ end of checkpoint
checkpoint = ModelCheckpoint(filepath='../weights/CRNN--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1)

# Save tensorboard log
NAME = "EMNIST-CRNN-Train-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='../logs/{}'.format(NAME))
 
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

# captures output of softmax so we can decode the output during visualization
model.fit_generator(generator=x_train.next_batch(),
                    steps_per_epoch=int(x_train.n / batch_size),
                    epochs=30,
                    callbacks=[checkpoint,tensorboard],
                    validation_data=x_val.next_batch(),
                    validation_steps=int(x_val.n / val_batch_size))

# END