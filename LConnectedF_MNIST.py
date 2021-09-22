""" Dropout layer randomly sets the input units to 0 with a frequency of rate at each step(helps prevent overfitting)
Use the resizing/rescaling/centercrop layer instead of resizing/rescaling/center cropping them in preprocessing.
Other options in Tensorboard callback are just for visualizing internal details of the model. """

#Imports
import os
import numpy
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf


#Constants:
LEARNING_RATE = 0.01
INPUT_SHAPE = (28,28,1)
EPOCHS = 2
CLASSES  = 10
BATCH_SIZE = 128
DROPOUT_RATE = 0.4
LOGS_DIR = "/tmp/tb/tf_logs/"

FILTER_1 = 4 ; KERNEL_1 = 3
FILTER_2 = 4 ; KERNEL_2 = 3
FILTER_3 = 4 ; KERNEL_3 = 3


#Callbacks.
tBoardCallback = tf.keras.callbacks.TensorBoard(LOGS_DIR,histogram_freq = 1, profile_batch = (500,520))


#DataLoading and Preprocessing
TrainImages,TrainLabels = numpy.load("Conv_Fashion_MNIST_TrainImages_0_1.npy"),numpy.load("Conv_Fashion_MNIST_TrainLabels_Categorical.npy")
TestImages,TestLabels   = numpy.load("Conv_Fashion_MNIST_TestImages_0_1.npy"),numpy.load("Conv_Fashion_MNIST_TestLabels_Categorical.npy")

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape = INPUT_SHAPE),

    tf.keras.layers.LocallyConnected2D(filters = FILTER_1,kernel_size = FILTER_1,),
    tf.keras.layers.MaxPooling2D((1,1)),
    tf.keras.layers.BatchNormalization(),

    tf.keras.layers.LocallyConnected2D(filters = FILTER_2,kernel_size = KERNEL_2,activation = "softplus"),
    tf.keras.layers.AveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    
    tf.keras.layers.LocallyConnected2D(filters = FILTER_3,kernel_size = KERNEL_3,activation = "relu"),
    tf.keras.layers.AveragePooling2D((1,1)),
    tf.keras.layers.Dropout(DROPOUT_RATE),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(CLASSES,activation = "softmax"),

])


#Model Compilation and Training.
model.compile(optimizer = tf.keras.optimizers.Adam(LEARNING_RATE),loss = "categorical_crossentropy",metrics = ["accuracy"])
model.fit(TrainImages,TrainLabels,epochs = EPOCHS,batch_size = BATCH_SIZE,callbacks = [tBoardCallback],validation_data = (TestImages,TestLabels))


# Model statistics and saving. 
model.save("SavedModels/Fashion_MNIST_LocallyConnected.h5")
model.save_weights("SavedModelWeights/Fashion_MNIST_LocallyConnected_Weights.h5")
