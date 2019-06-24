
import sys
sys.path.insert(1, './src')
from crfrnn_model2 import get_crfrnn_model_def
import util
import keras
import cv2
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing import image 
from sklearn.model_selection import train_test_split
import glob
import argparse
from keras.models import load_model
import random
import numpy as np
import glob
import itertools
from keras.layers.convolutional import Conv2D, ZeroPadding2D, UpSampling2D
from keras.layers.core import Flatten, Dense, Reshape, Permute, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import *
import os
#import args
import keras.models as Models
#from keras import models

from keras import backend as K
K.set_image_dim_ordering('th')

def getImageArr(path, width, height, imgNorm="sub_mean", odering='channels_first'):
    try:
        img = cv2.imread(path, 1)

        if imgNorm == "sub_and_divide":
            img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
        elif imgNorm == "sub_mean":
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
        elif imgNorm == "divide":
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img = img / 255.0

        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img
    except Exception as e:
        print(path, e)
        img = np.zeros((height, width, 3))
        if odering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img


def getSegmentationArr(path, nClasses, width, height):
    seg_labels = np.zeros((height, width, nClasses))
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (width, height))
        img = img[:, :, 0]

        for c in range(nClasses):
            seg_labels[:, :, c] = (img == c).astype(int)

    except Exception as e:
        print(e)
    seg_labels = np.reshape(seg_labels, (width * height, nClasses))
    return seg_labels


def imageSegmentationGenerator(images_path, segs_path, batch_size, n_classes, input_height, input_width, output_height,
                               output_width):
    #print('0')
    #assert images_path[-1] == '/'
    #assert segs_path[-1] == '/'
    #print('1')
    images = glob.glob(images_path + "*.jpg") 
    images.sort()
    #print(images)
    segmentations = glob.glob(segs_path + "*.jpg") + glob.glob(segs_path + "*.png") + glob.glob(segs_path + "*.jpeg")
    segmentations.sort()

    assert len(images) == len(segmentations)
    #print('ak')
    
#    for im, seg in zip(images, segmentations):
#      assert (im.split('/')[-1].split(".")[0] == seg.split('/')[-1].split(".")[0]
    zipped =itertools.cycle(zip(images, segmentations))

    while True:
        X = []
        Y = []
        for _ in range(batch_size):
            #print('akash')
            im,seg = next(zipped,(None,None))
            #print('akashyy')
            #print(im)
            #print(seg)
            X.append(getImageArr(im, input_width, input_height))
            Y.append(getSegmentationArr(seg, n_classes, output_width, output_height))

        yield np.array(X), np.array(Y)
        


def main():
	input_file = 'image.jpg'
	output_file = 'labels.png'

		    # Download the model from https://goo.gl/ciEYZi
	saved_model_path = 'crfrnn_keras_model.h5'
	model = get_crfrnn_model_def()
	parser = argparse.ArgumentParser()
	n_classes = args.n_classes
	#model_name = args.model_name
	images_path = args.test_images
	input_width =  args.input_width
	input_height = args.input_height
	epoch_number = args.epoch_number
	parser.add_argument("--epoch_number", type = int, default = 1 )
	parser.add_argument("--test_images", type = str , default = "/home/qwe/Downloads/leaf-image-segmentation-segnet-master/predict/0.png")
	parser.add_argument("--output_path", type = str , default = "/home/qwe/Downloads/leaf-image-segmentation-segnet-master/predictans")
	parser.add_argument("--input_height", type=int , default = 500  )
	parser.add_argument("--input_width", type=int , default = 500 )
	#parser.add_argument("--model_name", type = str , default = "vgg_segnet")
	parser.add_argument("--n_classes", type=int,default=3 )
	#print(sys.argv)
	parser.add_argument('--validate', action='store_true')
	#parser.add_argument("--model_name", type=str, default="vgg_segnet")
	parser.add_argument("--optimizer_name", type=str, default="adadelta")              
	args = parser.parse_known_args()[0]


	#m = modelFN(n_classes, input_height=input_height, input_width=input_width)
	m.compile(loss='categorical_crossentropy',
		  optimizer='adadelta',
		  metrics=['accuracy'])

	#if len(load_weights) > 0:
	#    m.load_weights(load_weights)

	print("Model output shape", m.output_shape)

	output_height = m.outputHeight
	output_width = m.outputWidth

	G = imageSegmentationGenerator('xtrain/', 'ytrain/', 8, n_classes,
		                                   input_height, input_width, output_height, output_width)

	#if validate:
	#    G2 = imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes, input_height,
	#                                                input_width, output_height, output_width)

	#if not validate:
	for ep in range(1):
		m.fit_generator(G, 100, epochs=1)
		#m.save_weights(save_weights_path + "." + str(ep))
		#m.save(save_weights_path + ".model." + str(ep))
	#else:
	#   for ep in range(epochs):
	#        m.fit_generator(G, 512, validation_data=G2, validation_steps=200, epochs=1)
	#        m.save_weights(save_weights_path + "." + str(ep))
	#        m.save(save_weights_path + ".model." + str(ep))

	    




	img_data, img_h, img_w = util.get_preprocessed_image(input_file)
	probs = model.predict(img_data, verbose=False)[0, :, :, :]
	segmentation = util.get_label_image(probs, img_h, img_w)
	segmentation.save(output_file)
'''
    train_generator = load_data_generator(xtrain, ytrain, batch_size=64)
    model.fit_generator(
	    generator=train_generator,
	    steps_per_epoch=900,
	    verbose=1,
	    epochs=5)
    test_generator = load_data_generator(xtest,ytest, batch_size=64)
    model.evaluate_generator(generator=test_generator,
                         steps=900,
                         verbose=1)
    model_name = "tf_serving_keras_mobilenetv2"
    model.save(f"models/{model_name}.h5")'''
if __name__ == '__main__':
    main()
