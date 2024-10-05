import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from collections import Counter
from math import sqrt

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Model, load_model, Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam,SGD
from matplotlib.image import imread
import tensorflow_addons as tfa 
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

class ImageGenerator:
    path_to_dataset=os.path.join('','dataset')
    image_size    = (224, 224, 3)
    batch_size    = 16  

    def __init__(self):
        self.class_names=None
        self.num_classes=-1
    
    def load_partial_images(self):
        training_set = tf.keras.utils.image_dataset_from_directory(directory = ImageGenerator.path_to_dataset,
                                                            validation_split = 0.2,
                                                            subset = "training",
                                                            seed = 1234,
                                                            batch_size = ImageGenerator.batch_size,
                                                            color_mode = 'rgb',
                                                            image_size = ImageGenerator.image_size[:len(ImageGenerator.image_size)-1],
                                                            interpolation = 'nearest',
                                                            crop_to_aspect_ratio = False,
                                                            shuffle = True)

        test_set= tf.keras.utils.image_dataset_from_directory(directory = ImageGenerator.path_to_dataset,
                                                            validation_split = 0.2,
                                                            subset = "validation",
                                                            seed = 1234,
                                                            batch_size = ImageGenerator.batch_size,
                                                            color_mode = 'rgb',
                                                            image_size = ImageGenerator.image_size[:len(ImageGenerator.image_size)-1], 
                                                            interpolation = 'nearest',
                                                            crop_to_aspect_ratio = False,
                                                            shuffle = True)

        
        self.num_classes = len(training_set.class_names)
        self.class_names = training_set.class_names
        images,labels = tuple(zip(*training_set))
        testimages,testlabels = tuple(zip(*test_set))
        xtrain,ytrain = tf.concat([image for image in images], 0), tf.concat([label for label in labels], 0)
        xtest,ytest = tf.concat([image for image in testimages], 0), tf.concat([label for label in testlabels], 0)
        return xtrain,xtest,ytrain,ytest

    def load_full_images(self):
        full_set = tf.keras.utils.image_dataset_from_directory(directory = ImageGenerator.path_to_dataset,
                                                            validation_split = None,
                                                            subset = None,
                                                            seed = 1234,
                                                            batch_size = ImageGenerator.batch_size,
                                                            color_mode = 'rgb',
                                                            image_size = ImageGenerator.image_size[:len(ImageGenerator.image_size)-1],
                                                            interpolation = 'nearest',
                                                            crop_to_aspect_ratio = False,
                                                            shuffle = True)
        self.num_classes = len(full_set.class_names)
        self.class_names = full_set.class_names
        images,labels = tuple(zip(*full_set))
        return tf.concat([image for image in images],0),tf.concat([label for label in labels],0)

    def display_dataset(self,dataset):
        for dimages, dlabels in dataset.take(1):
            for i in range(ImageGenerator.batch_size):
                plt.subplot(int(sqrt(ImageGenerator.batch_size)), int(sqrt(ImageGenerator.batch_size)), i+1)
                # Pixel values normalization for removing the negative values
                plt.imshow(tf.cast(dimages[i] /255, tf.float32))
                plt.title(self.class_names[dlabels[i]])
                plt.axis('off')
        plt.show()

class EpilepsySeqNet(Sequential):
    def __init__(self):
        super().__init__()
        self.image_generator = ImageGenerator()

    def load(self,k_fold=False):
        print(k_fold)
        if k_fold:
            xset,yset=self.image_generator.load_full_images()
        else:
            xtrain,xtest,ytrain,ytest=self.image_generator.load_partial_images()
        self.add(Input(shape=ImageGenerator.image_size, name='input_layer'))
        
        # Conv layer 1
        self.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding='same', name='conv_1')) 
        self.add(Activation(activation='relu', name='relu_1'))
        self.add(BatchNormalization(name='batch_norm_1'))
        self.add(MaxPooling2D(pool_size=(4,4), padding='valid', name='max_pool_1'))

        # Conv layer 2
        self.add(Conv2D(filters=96, kernel_size=(5,5), strides=(1,1), padding='same', name='conv_2')) 
        self.add(Activation(activation='relu', name='relu_2'))
        self.add(BatchNormalization(name='batch_norm_2'))
        self.add(MaxPooling2D(pool_size=(2,2), padding='valid', name='max_pool_2'))

        #Conv layer 3
        self.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding='same', name='conv_3')) 
        self.add(Activation(activation='relu', name='relu_3'))
        self.add(BatchNormalization(name='batch_norm_3'))
        self.add(MaxPooling2D(pool_size=(2,2), padding='valid', name='max_pool_3'))

        # Conv layer 4
        self.add(Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), padding='same', name='conv_4')) 
        self.add(Activation(activation='relu', name='relu_4'))
        self.add(BatchNormalization(name='batch_norm_4'))

        # Fully Connected layer
        self.add(Flatten(name='flatten'))
        print(Flatten)
        self.add(Dense(units=4096,activation='relu', name='dense_1'))
        self.add(Dense(units=2048,activation='relu', name='dense_2'))
        self.add(Dense(units=self.image_generator.num_classes, activation='softmax', name='output_layer'))

        self.compile(Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8),loss='sparse_categorical_crossentropy', metrics=['accuracy', tfa.metrics.F1Score(num_classes=self.image_generator.num_classes, average='micro', threshold=0.6)])

        if not k_fold:
            return xtrain,xtest,ytrain,ytest  
        else:
            return xset,yset

    def model_diagram(self):
        plt.figure(figsize=(10,10))
        plt.title("Epilypsis CNN network")
        plt.xticks([]), plt.yticks([])

        tf.keras.utils.plot_model(model=self, to_file=os.path.join('','EpilypsisSeqNet_schema.png'), show_shapes=True, show_layer_names=True)
        plt.imshow(imread(os.path.join('','EpilypsisSeqNet_schema.png')))
        plt.show()
        
    def clear_session(self):
        from tensorflow.python.framework import ops
        ops.reset_default_graph()
        tf.keras.backend.clear_session()

    def __str__(self):
        return str(self.summary())

def predict_with_train_test_split():
    model=EpilepsySeqNet()
    xtrain,xtest,ytrain,ytest=model.load(k_fold=False)

    callbacks=[
        ModelCheckpoint(filepath=os.path.join('','trained_models',f'epyllipsis.hdf5'), monitor='loss', mode='min', save_best_only=True, verbose=True),
        EarlyStopping(monitor='loss', mode='min', patience=4, min_delta=1e-3, restore_best_weights=True)
    ]
    
    model.fit(
        xtrain,
        ytrain,
        batch_size=ImageGenerator.batch_size,
        epochs=30,
        callbacks=callbacks,
        steps_per_epoch = xtrain.shape[0]//ImageGenerator.batch_size,
        verbose=True
    )

    y_probs=np.array(model.predict(xtest))
    y_preds=y_probs.argmax(axis=1)
    for i in range(ytest.shape[0]):
        print(f'True:{model.image_generator.class_names[ytest[i]]}\tPredicted:{model.image_generator.class_names[y_preds[i]]}')
    
    print('\n\n',classification_report(ytest,y_preds,target_names=model.image_generator.class_names))

def cross_validation():
    model=EpilepsySeqNet()
    xset,yset=model.load(k_fold=True)

    print(model.summary())
    xset,yset=np.array(xset),np.array(yset)

    cv_model=StratifiedKFold(n_splits=10,random_state=1234,shuffle=True)
    fold_no=1
    for train_indeces,test_indeces in cv_model.split(xset,yset):
        xtrain,xtest,ytrain,ytest=xset[train_indeces],xset[test_indeces],yset[train_indeces],yset[test_indeces]
        
        model.fit(
            xtrain,
            ytrain,
            epochs=30,
            callbacks=[
                ModelCheckpoint(filepath=os.path.join('','trained_models',f'epyllipsis_{fold_no}.hdf5'), monitor='loss', mode='min', save_best_only=True, verbose=True),
                EarlyStopping(monitor='loss', mode='min', patience=4, min_delta=1e-3, restore_best_weights=True)
            ],
            steps_per_epoch=xtrain.shape[0]//ImageGenerator.batch_size
        )
        fold_no+=1
        ypreds=np.array(model.predict(xtest)).argmax(axis=1)
        print(classification_report(ytest,ypreds,target_names=model.image_generator.class_names),end='\n\n')

        model.clear_session() # GPU clearence


if __name__ == '__main__':
    # predict_with_train_test_split() # Scenario 1-Predictions  made based on the train test split
    cross_validation() # Scenario 2-10 fold cross validation on epillipsis dataset