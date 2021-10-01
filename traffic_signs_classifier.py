import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import pandas as pd
import random
import cv2
from keras.preprocessing.image import ImageDataGenerator

import requests
from PIL import Image

np.random.seed(0)


url = "https://c8.alamy.com/comp/J2MRAJ/german-road-sign-bicycles-crossing-J2MRAJ.jpg"

with open("german-traffic-signs/train.p", mode="rb") as f:
    train_data = pickle.load(f)
with open("german-traffic-signs/valid.p", mode="rb") as f:
    val_data = pickle.load(f)
with open("german-traffic-signs/test.p", mode="rb") as f:
    test_data = pickle.load(f)

X_train, y_train = train_data["features"], train_data["labels"]
X_val, y_val = val_data["features"], val_data["labels"]
X_test, y_test = test_data["features"], test_data["labels"]

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

assert (
    X_train.shape[0] == y_train.shape[0]
), "The number of images is not equal to the number of labels."
assert (
    X_val.shape[0] == y_val.shape[0]
), "The number of images is not equal to the number of labels."
assert (
    X_test.shape[0] == y_test.shape[0]
), "The number of images is not equal to the number of labels."
assert X_train.shape[1:] == (
    32,
    32,
    3,
), "The dimensions of the images are not 32 x 32 x 3."
assert X_val.shape[1:] == (
    32,
    32,
    3,
), "The dimensions of the images are not 32 x 32 x 3."
assert X_test.shape[1:] == (
    32,
    32,
    3,
), "The dimensions of the images are not 32 x 32 x 3."

data = pd.read_csv("german-traffic-signs/signnames.csv")
num_of_samples = []
cols = 5
num_classes = 43

fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(5, 50))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(
            x_selected[random.randint(0, len(x_selected - 1)), :, :],
            cmap=plt.get_cmap("gray"),
        )
        axs[j][i].axis("off")
        # Center aligned Class index
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["SignName"])
            num_of_samples.append(len(x_selected))


print(num_of_samples)

plt.figure(figsize=(12, 4))
plt.bar(range(num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")


def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255
    return img


X_train = np.array(list(map(preprocessing, X_train)))
X_test = np.array(list(map(preprocessing, X_test)))
X_val = np.array(list(map(preprocessing, X_val)))

X_train = X_train.reshape(34799, 32, 32, 1)
X_test = X_test.reshape(12630, 32, 32, 1)
X_val = X_val.reshape(4410, 32, 32, 1)


datagen = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.3,
    shear_range=0.1,
    rotation_range=10,
)
datagen.fit(X_train)

batches = datagen.flow(X_train, y_train, batch_size=20)
X_batch, y_batch = next(batches)

fig, axs = plt.subplots(1, 15, figsize=(70, 10))
fig.tight_layout()
for i in range(15):
    axs[i].imshow(X_batch[i].reshape(32, 32), cmap="gray")
    axs[i].axis("off")

y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)
y_val = to_categorical(y_val, 43)


def modified_model():
    model = Sequential()
    model.add(Conv2D(60, (5, 5), input_shape=(32, 32, 1), activation="relu"))
    model.add(Conv2D(60, (5, 5), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(30, (3, 3), activation="relu"))
    model.add(Conv2D(30, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(
        Adam(learning_rate=0.0005),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = modified_model()
print(model.summary())

history = model.fit_generator(
    datagen.flow(X_train, y_train, batch_size=50),
    steps_per_epoch=X_train.shape[0] // 50,
    epochs=10,
    validation_data=(X_val, y_val),
    shuffle=1,
)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.legend(["training", "test"])
plt.title("Loss")
plt.xlabel("epoch")

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.legend(["training", "test"])
plt.title("Accuracy")
plt.xlabel("epoch")

score = model.evaluate(X_test, y_test, verbose=0)
print("Test score:", score[0])
print("Test accuracy:", score[1])


r = requests.get(url, stream=True)
img = Image.open(r.raw)
plt.imshow(img, cmap=plt.get_cmap("gray"))

img = np.asarray(img)
img = cv2.resize(img, (32, 32))
img = preprocessing(img)
plt.imshow(img, cmap=plt.get_cmap("gray"))
plt.axis("off")

img = img.reshape(1, 32, 32, 1)
print("predicted sign: " + str(model.predict_classes(img)))
