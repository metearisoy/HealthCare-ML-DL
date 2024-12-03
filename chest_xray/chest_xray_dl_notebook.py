# %% Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization 
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import cv2 # open cv
import os

from tqdm import tqdm


# %% Load Data

labels = ['PNEUMONIA', 'NORMAL']
img_size = 150  # 150 x 150

# Function to load and process training data
def get_training_data(data_dir):
    data = []  # To store images and labels
    for label in labels:
        path = os.path.join(data_dir, label)  # Path to each label folder
        class_num = labels.index(label)  # 0 for 'PNEUMONIA', 1 for 'NORMAL'

        # Ensure the folder exists
        if not os.path.exists(path):
            print(f"Directory not found: {path}")
            continue

        # Iterate through all images in the folder
        for img in tqdm(os.listdir(path)):
            try:
                # Read and preprocess the image
                img_path = os.path.join(path, img)
                img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img_arr is None:
                    print(f"Read image error: {img_path}")
                    continue

                resized_arr = cv2.resize(img_arr, (img_size, img_size))

                # Add data to the list
                data.append([resized_arr, class_num])
            except Exception as e:
                print(f"Error processing image {img}: {e}")
    
    return np.array(data, dtype=object)
        

# Load training data
train = get_training_data('D:/Healtcare-ML-DL/chest_xray/chest_xray/chest_xray/train')
test = get_training_data('D:/Healtcare-ML-DL/chest_xray/chest_xray/chest_xray/test')
val = get_training_data('D:/Healtcare-ML-DL/chest_xray/chest_xray/chest_xray/val')


# %% Data Visualization and Preprocessing

l = []

for i in train:
    if(i[1] == 0):
        l.append('PNEUMONIA')
    else:
        l.append('NORMAL')
        
sns.countplot(x=l)
    


x_train = []
y_train = []

x_test = []
y_test = []

for feature, label in train:
    x_train.append(feature)
    y_train.append(label)


for feature, label in test:
    x_test.append(feature)
    y_test.append(label)


plt.figure()
plt.imshow(train[0][0], cmap = 'gray')
plt.titel(labels[train[0][1]])


# Normalization [0, 1] ---- [0, 255] / 255 -> [0,1]

x_train = np.array(x_train)/255
x_test = np.array(x_test)/255

# (5126, 150, 150) -> (5126, 150, 150, 1)

x_train = x_train.reshape(-1, img_size, img_size, 1)
x_test = x_test.reshape(-1, img_size, img_size, 1)


y_train = np.array(y_train)
y_test = np.array(y_test)


# %% Data Augmentation

datagen = ImageDataGenerator(
    featurewise_center = False, # sets the global mean of the dataset to 0
    samplewise_center = False, # sets the mean of each sample to 0
    featurewise_std_normalization = False, # divides the data by the dataset's standard deviation,
    samplewise_std_normalization = False, # divides each sample by its own standard deviation,
    zca_whitening = False,  # applies ZCA whitening, reduces correlation,
    rotation_range = 30, # rotates images randomly by up to x degrees, 
    zoom_range = 0.2, # performs random zooming on images,
    width_shift_range = 0.1, # randomly shifts images horizontally,
    height_shift_range = 0.1, # randomly shifts images vertically,
    horizontal_flip = True, # randomly flips images horizontally,
    vertical_flip = True # randomly flips images vertically,
    )

datagen.fit(x_train)



# %% Create DL Model and Train

'''
Feature Exraction:
    con2d - Normalization - MaxPooling
    con2d - dropput - Normalization - MaxPooling
    con2d - Normalization - MaxPooling
    con2d - dropput - Normalization - MaxPooling
    con2d - dropput - Normalization - MaxPooling
    
Classification:
    flatten - Dense - Dropout - Dense (output)
Compiler: optimizer (rmsprop), Loss(binary cross ent.), metric (accuracy)
'''

model = Sequential()
model.add(Conv2D(128, (7,7), strides=1, padding='same', activation='relu', input_shape=(150, 150, 1)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))

model.add(Conv2D(64, (5, 5), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))

model.add(Conv2D(32, (3, 3), strides=1, padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))  # Binary classification

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose = 1, factor = 0.3, min_lr = 0.000001)
epoch_number = 3
history = model.fit(datagen.flow(x_train, y_train, batch_size = 32), epochs = epoch_number, validation_data = datagen.flow(x_test, y_test), callbacks = [learning_rate_reduction])
print('Loss of Model: ', model.evaluate(x_test, y_test)[0])
print('Accuracy of Model: ', model.evaluate(x_test, y_test)[1]*100)


'''
Overfitting Issue: The training accuracy (accuracy) is high, but the validation accuracy (val_accuracy) is significantly lower. T
his suggests that the model might be overfitting, meaning it has memorized the training data rather than generalizing well to unseen data.
-----
High Validation Loss: The validation loss (val_loss) is increasing steadily, indicating that the model is struggling on the validation data. T
o address this, you might consider adding more data, using regularization techniques (e.g., increasing the Dropout rate), or trying a different model architecture.
-----
Learning Rate Reduction: The learning rate has been reduced automatically (as shown by ReduceLROnPlateau), but the validation loss has not improved. 
This indicates that further fine-tuning might be necessary, such as training for more epochs with a lower learning rate.
'''
# %% Evaluatin

epochs = [i for i in range(epoch_number)]

a = history.history

fig, ax = plt.subplots(1,2)

train_acc = a['accuracy']
train_loss = a['loss']

val_acc = a['val_accuracy']
val_loss = a['val_loss']

ax[0].plot(epochs, train_acc, 'go-', label = 'training Accuracy')
ax[0].plot(epochs, val_acc, 'ro-', label = 'Validation Accuracy')
ax[0].legend()
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')

ax[1].plot(epochs, train_loss, 'go-', label = 'Training Loss')
ax[1].plot(epochs, val_loss, 'ro-', label = 'Validation Loss')
ax[1].legend()
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Loss')

