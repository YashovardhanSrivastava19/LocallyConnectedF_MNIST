import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']="true"
import numpy
import tensorflow as tf
(xTrain,yTrain),(xTest,yTest) = tf.keras.datasets.fashion_mnist.load_data()

xTrain = xTrain.reshape(xTrain.shape[0],28,28,1).astype('float32')/255
xTest = xTest.reshape(xTest.shape[0],28,28,1).astype('float32')/255

yTrain = tf.keras.utils.to_categorical(yTrain,10)
yTest = tf.keras.utils.to_categorical(yTest,10)

numpy.save("Conv_Fashion_MNIST_TrainImages_0_1",xTrain)
numpy.save("Conv_Fashion_MNIST_TrainLabels_Categorical",yTrain)
numpy.save("Conv_Fashion_MNIST_TestImages_0_1",xTest)
numpy.save("Conv_Fashion_MNIST_TestLabels_Categorical",yTest)
