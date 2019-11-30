import tensorflow as tf

from tensorflow_core.python.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow_core.python.keras.optimizer_v2.adam import Adam
from tensorflow_core.python.keras import datasets
import matplotlib.pyplot as plt

import numpy as np
import cv2 as cv
import os
from imutils import build_montages

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Data preparation
mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Create CNN Model

NUM_EPOCHS = 25
LR = 0.001
BATCH_SIZE = 32

model = tf.keras.Sequential()
model.add(Conv2D(28, (3, 3), strides=2, padding='same', activation='relu', input_shape=(28, 28, 1)))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="best_weights.hdf5",
                                                  monitor='val_accuracy',
                                                  verbose=1,
                                                  save_best_only=True)

opt = Adam(lr=LR, decay=LR / NUM_EPOCHS)

model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train[:, :, :, np.newaxis],
                    y_train, epochs=NUM_EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[checkpointer],
                    validation_split=0.2)

model.save('cnn.h5')
# model.load_weights('best_weights.hdf5')
# Plot learning curve

test_loss, test_acc = model.evaluate(x_test[:, :, :, np.newaxis], y_test, verbose=False)
print('Tested Acc: ', test_acc)
print('Tested Loss: ', test_loss)

fig4, ax4 = plt.subplots(nrows=2, ncols=1, figsize=(10
                                                    , 10))
# Plot training & validation accuracy values
ax4[0].plot(history.history['accuracy'])
ax4[0].plot(history.history['val_accuracy'])
ax4[0].set_title('Model accuracy')
ax4[0].set(ylabel='Accuracy', xlabel='Epoch')
ax4[0].legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
ax4[1].plot(history.history['loss'])
ax4[1].plot(history.history['val_loss'])
ax4[1].set_title('Model loss')
ax4[1].set(ylabel='Loss', xlabel='Epoch')
ax4[1].legend(['Train', 'Validation'], loc='upper left')
# Prediction

predictions = model.predict(x_test[:, :, :, np.newaxis])
predicted_args = np.argmax(predictions, axis=1)

i = 0

fig1, ax1 = plt.subplots()
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set(xlabel='predicted: {}, label: {}'.format(class_names[predicted_args[i]], class_names[y_test[i]]))
ax1.imshow(x_test[i], cmap=plt.cm.binary)

# Plot activations of the first layer

layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(x_test[:1000, :, :, np.newaxis])

first_layer_activation = activations[0]
print(first_layer_activation.shape)

fig2, ax2 = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
for axis, i in zip(ax2.ravel(), range(16)):
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    axis.imshow(activations[0][0, :, :, i], cmap=plt.cm.binary)

# Fashion MNIST classification

# initialize list of output images
images = []

random_choice = np.random.choice(np.arange(0, len(y_test)), size=(16,))
for i in random_choice:

    image = (x_test[i] * 255).astype("uint8")

    color = (0, 255, 0)

    if predicted_args[i] != y_test[i]:
        color = (255, 0, 0)

    image = cv.merge([image] * 3)
    image = cv.resize(image, (96, 96), interpolation=cv.INTER_LINEAR)
    cv.putText(image, class_names[y_test[i]], (5, 20), cv.FONT_HERSHEY_SIMPLEX, 0.75,
               color, 2)

    images.append(image)

montage = build_montages(images, (96, 96), (4, 4))[0]

fig3, ax3 = plt.subplots()
ax3.set_xticks([])
ax3.set_yticks([])
ax3.imshow(montage, cmap=plt.cm.binary)

plt.show()
