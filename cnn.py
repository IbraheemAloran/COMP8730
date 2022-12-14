

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

#parameters
learningRate = 0.01
batch = 64
epochs = 10
imgSize = (28,28,1)
classes = 10

#load the dataset and split into train and test sets and print the shape
(trainData, trainlabels), (testData, testlabels) = keras.datasets.mnist.load_data()

print("Train images shape:", trainData.shape)
print("Test images shape:", testData.shape)
print("Train labels shape:", trainlabels.shape)
print("Test labels shape:", testlabels.shape)


#transform the label data to a binary matrix so that it can work with the loss function
trainlabels = keras.utils.to_categorical(trainlabels, classes)
testlabels = keras.utils.to_categorical(testlabels, classes)

#Show the first image of the dataset
plt.imshow(trainData[0], cmap='gray')
plt.show()

#define the CNN and print the summary
model = keras.Sequential([
    keras.Input(shape=imgSize),

    # Here we normalize the input image and rescale the rgb values from the range [0,255] to [0,1]
    keras.layers.Rescaling(scale=1./255, input_shape=(imgSize)), 

    #first convolutional block
    layers.Conv2D(16, kernel_size=(3, 3), strides=(1,1), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    #second convolutional block
    layers.Conv2D(32, kernel_size=(3, 3), strides=(1,1), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    #third convolutional block
    layers.Conv2D(64, kernel_size=(3, 3), strides=(1,1), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),

    #Fully connected layer
    layers.Flatten(),
    layers.Dropout(0.2),

    #Output Layer
    layers.Dense(classes, activation='softmax')
    
])

model.summary()

#define the loss function and optimizer with the learning rate
optim = keras.optimizers.Adam(learning_rate=learningRate)
lossFn = tf.keras.losses.CategoricalCrossentropy() #we use CategoricalCrossEntropy for the loss because we have 10 classes

#Compile and Train the model
model.compile(loss=lossFn, optimizer=optim, metrics=["accuracy"])
stats = model.fit(trainData, trainlabels, batch_size=batch, epochs=epochs, validation_split=0.1)

#Evaluate the model on the test data
results = model.evaluate(testData, testlabels)
print("ACCURACY:", results[1])

#Graph the losses and accuracies
plt.plot(range(epochs), stats.history["loss"])
plt.plot(range(epochs), stats.history["val_loss"])
plt.legend(["Training Loss", "Validation Loss"])
plt.title("Training and Validation Loss")
plt.show()
plt.plot(range(epochs), stats.history["accuracy"])
plt.plot(range(epochs), stats.history["val_accuracy"])
plt.legend(["Training accuracy", "Validation accuracy"])
plt.title("Training and Validation Accuracy")
plt.show()
