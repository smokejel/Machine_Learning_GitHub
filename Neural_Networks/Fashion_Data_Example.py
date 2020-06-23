import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def data_summary(dataset):
    # shape
    print(dataset.shape)
    print(dataset[0,23,23])
    plt.figure()
    plt.imshow(dataset[5])
    plt.colorbar()
    plt.grid(False)
    plt.show()

'''Load dataset'''
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# data_summary(train_images)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''Data Preprocessing
In this case we will simply scale all our greyscale pixel values (0-255) to be between 0 and 1.
We can do this by dividing each value in the training and testing sets by 255.0.
We do this because smaller values will make it easier for the model to process our values.'''
train_images = train_images / 255.0
test_images = test_images / 255.0

'''Build the Model
1. Sequential Model used with three different layers
2. Input layer will consist of a vector of 784 (28x28) neurons. The flatten function used to 
   transform 28x28 input into 784 vector.
3. Hidden layer: The *dense* denotes that this layer will be fully connected and each neuron 
   from the previous layer connects to each neuron of this layer. 
   It has 128 neurons and uses the rectify linear unit activation function.
4. This is our output later and is also a dense layer. It has 10 neurons that we will 
   look at to determine our models output. Each neuron represnts the probabillity of a given 
   image being one of the 10 different classes. The activation function *softmax* is used on 
   this layer to calculate a probabillity distribution for each class. 
   This means the value of any neuron in this layer will be between 0 and 1, 
   where 1 represents a high probabillity of the image being that class.'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer
    keras.layers.Dense(128, activation='relu'),  # hidden layer
    keras.layers.Dense(10, activation='softmax') # output layer
])

'''Compile Model
The last step in building the model is to define the loss function, 
optimizer and metrics we would like to track.'''
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

'''Training the Model
Simply call a fit method'''
model.fit(train_images, train_labels, epochs=10)

'''Evaluating the Model
The verbose argument is defined from the keras documentation as: 
"verbose: 0 or 1. Verbosity mode. 0 = silent, 1 = progress bar."
https://keras.io/models/sequential/'''
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)
print('Test accuracy:', test_acc)

