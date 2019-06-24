Installation Guide

1.1 Install dependencies

Install Tensorflow and Keras, following the respective installation guides. You will need to install Keras with HDF5/h5py if you plan to use the provided trained model. After installing these two packages, run the following commands to make sure they are properly installed:

# First, activate the correct Python virtualenv if you used one during Tensorflow/Keras installation

$ source /home/user/tensorflow_virtualenv/bin/activate  

$ python

>>> import tensorflow
>>> import keras
You should not see any errors while importing tensorflow and keras above.

1.2 Build CRF-RNN custom C++ code

Checkout the code in this repository, activate the Tensorflow/Keras virtualenv (if you used one), and run the compile.sh script in the cpp directory. That is, run the following commands:

$ git clone https://github.com/sadeepj/crfasrnn_keras.git

$ cd crfasrnn_keras/cpp

$ source /home/user/tensorflow_virtualenv/bin/activate

$ ./compile.sh

If the build succeeds, you will see a new file named high_dim_filter.so. If it fails, please see the comments inside the compile.sh file for help. You could also refer to the official Tensorflow guide for building a custom op.

Note: This script will not work on Windows OS. If you are on Windows, please check this issue and the comments therein. The official Tensorflow guide for building a custom op does not yet include build instructions for Windows.

1.3 Download the pre-trained model weights

Download the model weights from here and place it in the crfasrnn_keras directory with the file name crfrnn_keras_model.h5.

1.4 Run the demo

$ cd crfasrnn_keras

$ python run_demo.py  # Make sure that the correct virtualenv is already activated

If everything goes well, you will see the segmentation results in a file named "labels.png"
