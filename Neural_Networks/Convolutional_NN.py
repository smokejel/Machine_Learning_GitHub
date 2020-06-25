import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

#  LOAD AND SPLIT DATASET
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

'''# Let's look at a one image
IMG_INDEX = 7  # change this to look at other images
plt.imshow(train_images[IMG_INDEX] ,cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()'''

# Define Model
'''The input shape of our data will be 32, 32, 3 and we will process 32 filters of size 3x3 over our input data. 
We will also apply the activation function relu to the output of each convolution operation.'''
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

'''This layer will perform the max pooling operation using 2x2 samples and a stride of 2.'''
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

'''The next set of layers do very similar things but take as input the feature map from the previous layer. 
They also increase the frequency of filters from 32 to 64. We can do this as our data shrinks in spacial 
dimensions as it passed through the layers, meaning we can afford (computationally) to add more depth.'''
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Adding Dense Layers. These layers are used to classify data
model.add(layers.Flatten())  # transform 4x4x64 array into a vector
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()  # let's have a look at our model so far

# Training the Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# Evaluate the Model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)

'''Data Augmentation 
We can create new test data by performing random transformations on our images so that our model can 
generalize/perform better. '''
# creates a data generator object that transforms images
datagen = ImageDataGenerator(rotation_range=40,
                             width_shift_range=40,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode='nearest')

# pick an image to transform
test_img = train_images[14]
img = image.img_to_array(test_img) # Convert image to numpy array
img = img.reshape((1,) + img.shape)

i = 0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):  # this loops runs forever until we break, saving images to current directory with specified prefix
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i > 4:  # show 4 images
        break

plt.show()
