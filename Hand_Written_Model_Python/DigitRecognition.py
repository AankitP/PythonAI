import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

######## This was used to train and save the model ###########

# # Grab the mnist dataset which is already separated
# mnist = tf.keras.datasets.mnist

# # X data is the pixel data (image)
# # Y data is the actual number X is representing
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# # to normalize values from 0 to 1
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# # to create the neural network model
# model = tf.keras.models.Sequential()

# # to add layers to the model
# # Flatten layer turns the image into a big line of 784 pixels
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# # basic neural network layer, where each neuron is connected 
# # to each neuron of other layers
# AF = 'tanh' # This is the type of activation function I want to use
# model.add(tf.keras.layers.Dense(128, activation=AF))
#     # for activation, you can also put it in formaat: 
#         # tf.nn.<Activation function
#     # relu means rectify linear unit
# model.add(tf.keras.layers.Dense(128, activation=AF))
#     # to add amother layer
# model.add(tf.keras.layers.Dense(10, activation='softmax'))
#     # This is to make this layer represent 10 digits (0-9)
#     # activation function is softmax to make sure that all
#     #  the outputs add up to 1, kind of like a confidenct level

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # to train the model
# model.fit(x_train, y_train, epochs=20)
#     # epoch is the amount of times that the model will 
#     # see the data repeatedly

# model.save('HW_model')

######## Using the saved model on the test data ###########

# model = tf.keras.models.load_model('HW_model')

# loss, accuracy = model.evaluate(x_test, y_test)

# print(loss)
# print(accuracy)

######### Using my own digit drawings #############

model = tf.keras.models.load_model('HW_model')

image_number = 1

# check that the files exist
print(os.path.isfile(f"drawn_number_images/{image_number}.png"))
while os.path.isfile(f"drawn_number_images/{image_number}.png"):
    try:
        # import and read the images
        img = cv2.imread(f"drawn_number_images/{image_number}.png")[:,:,0]
        
        # Invert the images, and make the image an array
        img = np.invert(np.array([img]))
        
        # Get the model to guess what number it is
        prediction= model.predict(img)
        
        # print the model's prediction
        # np.argmax gives the index of the field 
        # that has the heighest number to give us
        # the most probable digit
        print(f"This digit is probably a {np.argmax(prediction)}")
        
        # show the image
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

        print(image_number)
    
    except:
        # if there is an error print error
        print("error")
    
    finally:
        # make sure that the image number iterates
        image_number += 1