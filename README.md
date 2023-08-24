# Anomaly Detection Using Convolutional Autoencoder
The project implementation of a convolutional autoencoder as a TensorFlow Keras Model provides a flexible and customizable way to define and train a convolutional autoencoder for anomaly detection tasks.

# Dataset
The model is trained on a Cifar-10 dataset of normal images and then tested on a dataset containing both normal and anomalous images. While loading the images, added a data augmentation class. This will improve the generalization ability.
-  load cifar10 this will load the CIFAR-10 dataset with the Tensorflow built-in function. It will generate and shuffle CIFAR-10 iterators using ”tf.data.Data”
-  Inverse specific labeled images: This will invert the images with a single specified label.
-  inverse multiple labeled images: This will invert the images with a list of specified labels.

# Structure of Code
First, import the libraries, including TensorFlow, matplot, and NumPy.
Define the parameters of the autoencoder, including the number of neurons in the first fully connected layer (num neuron), the number of filters in the first and second convolutional layers (kernal1 and kernal2), and the shape of the input images.
-  First, import the libraries, including TensorFlow, matplot and NumPy.
-  Define the parameters of the autoencoder, including the number of neurons in the first fully connected layer (num neuron), the numbers of filters in the first and second convolutional layers (kernal1 and kernal2), and the shape of the input images.
-  Next, Define the "Convolutional Autoencoder" class, this will extends the tf.keras.Model class.
-  The init method set up the layers of the autoencoder.
-  Next, define a function load data that loads CIFAR-10 dataset and preprocesses the training and test datasets.
-  Next, define a function load data that loads CIFAR-10 dataset and preprocesses the training and test datasets.
-  Next will create an instance of the ConvolutionalAutoencoder class and will compile it with an optimizer and loss function, and fit it to the training data.
-  Next will calculate the performance of the autoencoder on the test data.
-  Finally, use the autoencoder to detect anomalies in input or test images by calculating the reconstruction error and comparing it to a threshold value.

# Huggingface Demo 
[Demo - Anomaly Detection using Convolutional Autoencoder](https://huggingface.co/spaces/Charmi06/Anomaly)

