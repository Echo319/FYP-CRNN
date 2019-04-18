Download and unpack https://www.kaggle.com/crawford/emnist dataset into ./Dataset

HOWTO: 
CD into source
run "python GenerateWords.py" - Can change the number of instances in the script 
run "python Train.py" 
Pick the best weights from ./weights place in root directory and rename to final_weights.hdf5
run "python predict.py"

FILES:
Dataset
        - Words
            - Train          Training data
            - Test           Test data
            - Predict        Used in predict.py
        - EMNSIT Dataset
docs    
        - Stuff for report
logs    - location for tensorboard
        - EMNIST-CRNN-Train-xxxxxxx 
            -- Log data for tensorboard
src
        - DataGenerator.py     Keras batch generator used in Train
        - GenerateWords.py     Uses EMNIST dataset to create random words put in ../Dataset/words
        - Model.py             The definition for the CRNN model
        - predict.py           Tester reports accuracy
        - Train.py             Run this to train the model, weights are saved at the end of every epoch in ../weights
weights
        - CRNN-XX-XXXX.hdf5    A collection of weights saved from every run. pull the best out and rename to final_wights.hdf5 to use in predict.py

README.txt      This file
