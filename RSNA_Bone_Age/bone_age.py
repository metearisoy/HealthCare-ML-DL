# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread

import os

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings("ignore")


# %% load dataset:train, test
dataframe_direction = "bonage_dataset"
training_dataframe = pd.read_csv("bonage_dataset/boneage-training-dataset.csv")

training_dataframe["path"] = training_dataframe["id"].map(lambda x: os.path.join(dataframe_direction,"boneage-training-dataset", "boneage-training-dataset","{}.png".format(x)))
# bonage_dataset/boneage-training-dataset/boneage-training-dataset/1377.png

training_dataframe["imagepath"] = training_dataframe["id"].map(lambda x: "{}.png".format(x))
training_dataframe["gender"] = training_dataframe["male"].map(lambda x: "male" if x else "female")
training_dataframe["gender_encoded"] = training_dataframe["gender"].map(lambda x: 1 if x == "male" else 0)
training_dataframe["boneage_category"] = pd.cut(training_dataframe["boneage"],10)

boneage_std = 2*training_dataframe["boneage"].std()
boneage_mean = training_dataframe["boneage"].mean()
training_dataframe["norm_age"] = (training_dataframe["boneage"] - boneage_mean)/boneage_std

# test
dataframe_direction = "bonage_dataset"
testing_dataframe = pd.read_csv("bonage_dataset/boneage-test-dataset.csv")

testing_dataframe["path"] = testing_dataframe["Case ID"].map(lambda x: os.path.join(dataframe_direction,"boneage-test-dataset", "boneage-test-dataset","{}.png".format(x)))
# bonage_dataset/boneage-training-dataset/boneage-training-dataset/1377.png

testing_dataframe["imagepath"] = testing_dataframe["Case ID"].map(lambda x: "{}.png".format(x))
testing_dataframe["gender"] = testing_dataframe["Sex"].map(lambda x: "male" if x=="M" else "female")
testing_dataframe["gender_encoded"] = testing_dataframe["Sex"].map(lambda x: 1 if x == "male" else 0)

# visualization
single_img = training_dataframe["path"][15]
imgs = imread(single_img)
plt.imshow(imgs, cmap="gray")

# %% preprocessing: train test split, data augmentation 

df_train, df_val = train_test_split(training_dataframe, test_size = 0.2, random_state=42, shuffle=True)
valid_df, df_test = train_test_split(df_val, test_size=0.5, random_state=42, shuffle=True)

data_augmentation = dict(rotation_range=180,
                         zoom_range=0.25,
                         brightness_range=[0.2,0.5],
                         width_shift_range = 0.2,
                         height_shift_range = 0.2,
                         horizontal_flip = True,
                         shear_range = 0.05,
                         fill_mode = "nearest")

train_generator = ImageDataGenerator(rescale = 1/255, preprocessing_function = preprocess_input, **data_augmentation)
test_val_generator = ImageDataGenerator(rescale = 1/255, preprocessing_function = preprocess_input)

img_size = (256, 256)
batch_size = 32

train_data = train_generator.flow_from_dataframe(
                dataframe = df_train,
                x_col = "path",
                y_col = "boneage",
                batch_size=batch_size,
                seed=42,
                shuffle=True,
                class_mode="other",
                flip_vertical = True,
                color_mode = "rgb",
                target_size = img_size)

valid_data = test_val_generator.flow_from_dataframe(
                dataframe = valid_df,
                x_col = "path",
                y_col = "boneage",
                batch_size=batch_size,
                seed=42,
                shuffle=False,
                class_mode="other",
                flip_vertical=True,
                color_mode = "rgb",
                target_size=img_size)

test_data = test_val_generator.flow_from_dataframe(
                dataframe = df_test,
                x_col = "path",
                y_col = "boneage",
                batch_size=batch_size,
                seed=42,
                shuffle=False,
                class_mode="other",
                flip_vertical=True,
                color_mode = "rgb",
                target_size=img_size)

X_test, y_test = next(test_val_generator.flow_from_dataframe(
                        df_test,
                        x_col = "path",
                        y_col = "boneage",
                        batch_size=10*batch_size,
                        class_mode="other",
                        flip_vertical=True,
                        color_mode = "rgb",
                        target_size=img_size))

# %% Transfer learning: Xception -> fine tuning
model = tf.keras.applications.xception.Xception(input_shape=(256,256,3), 
                                                include_top=False, 
                                                weights="imagenet")

model.trainable = True
new_model = Sequential()
new_model.add(model)
new_model.add(GlobalMaxPooling2D())
new_model.add(Flatten())
new_model.add(Dense(5))
new_model.add(Activation("relu"))
new_model.add(Dense(1, activation="linear"))

new_model.compile(loss = "mse", optimizer= Adam(learning_rate=0.001), metrics = ["mse"])
callback = EarlyStopping(monitor= "val_loss", restore_best_weights = True)
history = new_model.fit(train_data, epochs=2, validation_data=valid_data, batch_size=batch_size, callbacks = [callback])



# %% model evaluation

y_pred = new_model.predict(X_test, batch_size=64, verbose = 1)

ord_ = np.argsort(y_test)
ord_ = np.array([1,45,67,89,91,109])

fig, ax = plt.subplots(1, 6, figsize=(20,20))

for (i, ax) in zip(ord_, ax.flatten()):
    ax.imshow(X_test[i, :,:,0],cmap="gray")
    ax.set_title(f"Age: {y_test[i]}, \n Predicted: {y_pred[i][0]}")
    ax.axis("off")
    
plt.tight_layout()
plt.show()








































