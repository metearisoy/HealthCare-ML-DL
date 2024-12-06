# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model
from tensorflow.keras.layers.experimental import preprocessing

from pathlib import Path
import os.path

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


# %% load data
dataset = "Pharmaceutical_Drugs_Vitamins_Synthetic/Data Combined"
image_dir = Path(dataset)

filepaths = list(image_dir.glob(r"**/*.jpg")) + list(image_dir.glob(r"**/*.png"))

labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

filepaths = pd.Series(filepaths, name = "filepath").astype(str)
labels = pd.Series(labels, name="label")

image_df = pd.concat([filepaths, labels], axis=1)

# %% data visualization
random_index = np.random.randint(0, len(image_df), 25)
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(11,11))

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.filepath[random_index[i]]))
    ax.set_title(image_df.label[random_index[i]])
plt.tight_layout()


# %% data preprocessing: train-test split, data augmentation, resize, rescaling

# train-test split
train_df, test_df = train_test_split(image_df, test_size=0.2, shuffle=True, random_state=42)

# data augmentation
train_generator = ImageDataGenerator(
                    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
                    validation_split=0.2)

test_generator = ImageDataGenerator(
                    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)

train_images = train_generator.flow_from_dataframe(
                        dataframe=train_df,
                        x_col = "filepath", # independent -> image
                        y_col = "label", # dependent -> target variable -> label
                        target_size = (224,224), #  size of the images
                        color_mode = "rgb",
                        class_mode = "categorical",
                        batch_size=64,
                        shuffle=True,
                        seed=42,
                        subset="training"
                        ) 

val_images = train_generator.flow_from_dataframe(
                    dataframe=train_df,
                    x_col="filepath",
                    y_col="label",
                    target_size=(224,224),
                    color_mode = "rgb",
                    class_mode="categorical",
                    batch_size=64,
                    shuffle=True,
                    seed=42,
                    subset="validation")

test_images = test_generator.flow_from_dataframe(
                    dataframe=test_df,
                    x_col="filepath",
                    y_col="label",
                    target_size=(224,224),
                    color_mode="rgb",
                    class_mode="categorical",
                    batch_size=64,
                    shuffle = False)

# resize, rescale
resize_and_rescale = tf.keras.Sequential(
    [
     layers.experimental.preprocessing.Resizing(224, 224),
     layers.experimental.preprocessing.Rescaling(1./255)
     ])

# %% transfer learning modeli (MobileNetV2), training
pretrained_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),  # The input shape, i.e., the size of the input images (224x224 pixels with 3 color channels for RGB)
    include_top=False,          # Excludes the top classification layer of MobileNet (set to False to add custom layers for classification)
    weights="imagenet",         # Specifies the pre-trained weights, trained on the ImageNet dataset
    pooling="avg"               # Global average pooling will be applied after the last convolutional layer
)

pretrained_model.trainable = False

# create checkpoint callback
checkpoint_path = "pharmaceutical_drugs_and_vitamins_classification_model_checkpoint"
checkpoint_callback = ModelCheckpoint(checkpoint_path,
                                      save_weights_only = True,
                                      monitor="val_accuracy",
                                      save_best_only=True)

early_stopping = EarlyStopping(monitor="val_loss",
                               patience = 5,
                               restore_best_weights = True)

# training model - classification blok
inputs = pretrained_model.input
x = resize_and_rescale(inputs)
x = Dense(256, activation="relu")(pretrained_model.output)
x = Dropout(0.2)(x)
x = Dense(256, activation = "relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(10, activation="softmax")(x)
model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer=Adam(0.0001), loss="categorical_crossentropy", metrics = ["accuracy"])

history = model.fit(
    train_images,
    steps_per_epoch = len(train_images),
    validation_data = val_images,
    validation_steps=len(val_images),
    epochs=10,
    callbacks = [early_stopping,checkpoint_callback])

# %% model evaluation

results = model.evaluate(test_images, verbose = 1)

print("Test loss: ",results[0])
print("Test accuracy: ",results[1])

epochs = range(1, len(history.history["accuracy"]) + 1)
hist = history.history

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(epochs, hist["accuracy"], "bo-", label = "Training Accuracy")
plt.plot(epochs, hist["val_accuracy"], "r^-", label = "Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, hist["loss"], "bo-", label = "Training Loss")
plt.plot(epochs, hist["val_loss"], "r^-", label = "Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 
pred = model.predict(test_images)
pred = np.argmax(pred, axis=1)

labels = (train_images.class_indices)
labels = dict((v,k) for k,v in labels.items())
pred = [labels[k] for k in pred]

random_index = np.random.randint(0, len(test_df) - 1, 15)
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(11,11))

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.filepath[random_index[i]]))
    if test_df.label.iloc[random_index[i]] == pred[random_index[i]]:
        color = "green"
    else:
        color = "red"
    ax.set_title(f"True: {test_df.label.iloc[random_index[i]]}\n predicted: {pred[random_index[i]]}", color = color)
plt.tight_layout()

y_test = list(test_df.label)
print(classification_report(y_test, pred))