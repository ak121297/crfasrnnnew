

import sys
sys.path.insert(1, './src')
from crfrnn_model import get_crfrnn_model_def
import util
import keras
#import xtrain 
#import ytrain
import cv2
from keras.preprocessing.image import ImageDataGenerator
#from keras.preprocessing import image 
from sklearn.model_selection import train_test_split
import glob
def main():
    input_file = 'image.jpg'
    output_file = 'labels.png'

    # Download the model from https://goo.gl/ciEYZi
    saved_model_path = 'crfrnn_keras_model.h5'

    model = get_crfrnn_model_def()

    from keras.optimizers import Adam
    #model = build_model()
    model.compile(optimizer=Adam(),
		      loss='categorical_crossentropy',
		      metrics=['categorical_accuracy'])
    
    # we create two instances with the same arguments
    data_gen_args = dict(featurewise_center=True,
                     featurewise_std_normalization=True,
                     rotation_range=90,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
    image_datagen = keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    mask_datagen =  keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    #xtrain=/home/qwe/Downloads/crfrnn2/xtrain
    #ytrain='/home/qwe/Downloads/crfrnn2/ytrain'
    #grayx=cv2.cvtColor(xtrain,cv2.COLOR_BGR2GRAY)
    pathx='/home/qwe/Downloads/crfrnn2/xtrain/*.*'
    pathy='/home/qwe/Downloads/crfrnn2/ytrain/*.*'
    for file in glob.glob(pathx):
         c1=cv2.imread(file,0)
       # c1=cv2.cvtColor(a1,cv2.COLOR_BGR2RGB)
    for file in glob.glob(pathy):
         c2=cv2.imread(file,0)
       # c2=cv2.cvtColor(a2,cv2.COLOR_BGR2RGB)
    
    #xtrain=cv2.imread('/home/qwe/Downloads/crfrnn2/xtrain',0)
    #ytrain=cv2.imread('/home/qwe/Downloads/crfrnn2/ytrain',0)
    #grayx=cv2.cvtColor(xtrain,cv2.COLOR_BGR2GRAY)
    #grayY=cv2.cvtColor(Ytrain,cv2.COLOR_BGR2GRAY)
    
    #print(xtrain)
    image_datagen.fit(c1)
    mask_datagen.fit(c2, augment=False, seed=seed)

    image_generator = image_datagen.flow_from_directory(
    'home/qwe/Downloads/xtrain',
    class_mode=None,
    seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
    'home/qwe/Downloads/ytrain',
    class_mode=None,
    seed=seed)

    # combine generators into one which yields image and masks
    train_generator = zip(image_generator, mask_generator)

    model.fit_generator(
    train_generator,
    steps_per_epoch=2000,
    epochs=50)
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

    #model.load_weights(saved_model_path)



if __name__ == '__main__':
    main()
