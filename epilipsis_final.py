from random import random
from tabnanny import verbose
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import cv2
from matplotlib.gridspec import GridSpec
from collections import Counter
from math import sqrt
from skimage.morphology import skeletonize

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
from tensorflow.python.framework import ops

from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score    

class ImageGenerator:
    path_to_dataset = os.path.join('','dataset')
    image_size    = (224, 224, 3)
    batch_size    = 16  

    def __init__(self):
        self.class_names = None
        self.num_classes = -1
    
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
        images, labels = tuple(zip(*training_set))
        testimages, testlabels = tuple(zip(*test_set))
        xtrain, ytrain = tf.concat([image for image in images], 0), tf.concat([label for label in labels], 0)
        xtest, ytest = tf.concat([image for image in testimages], 0), tf.concat([label for label in testlabels], 0)
        return xtrain, xtest, ytrain, ytest

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
        images, labels = tuple(zip(*full_set))
        return tf.concat([image for image in images],0), tf.concat([label for label in labels],0)

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
    callbacks=[
        ModelCheckpoint(filepath=os.path.join('','trained_models', f'epyllipsis.hdf5'), monitor='loss', mode='min', save_best_only=True, verbose=True),
        EarlyStopping(monitor='loss', mode='min', patience=4, min_delta=1e-3, restore_best_weights=True)
    ]

    def __init__(self):
        super().__init__()
        self.image_generator = ImageGenerator()

    def load(self, k_fold=False):
        print(k_fold)
        if k_fold:
            xset, yset = self.image_generator.load_full_images()
        else:
            xtrain, xtest, ytrain, ytest = self.image_generator.load_partial_images()

        self.add(Input(shape=ImageGenerator.image_size, name='input_layer'))
        
        # Conv layer 1
        self.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding='same', name='conv_1')) 
        self.add(Activation(activation='relu', name='relu_1'))
        self.add(BatchNormalization(name='batch_norm_1'))
        self.add(MaxPooling2D(pool_size=(2,2), padding='valid', name='max_pool_1')) # 112

        # Conv layer 2
        self.add(Conv2D(filters=96, kernel_size=(5,5), strides=(1,1), padding='same', name='conv_2')) 
        self.add(Activation(activation='relu', name='relu_2'))
        self.add(BatchNormalization(name='batch_norm_2'))
        self.add(MaxPooling2D(pool_size=(2,2), padding='valid', name='max_pool_2')) # Pooling->56

        #Conv layer 3
        self.add(Conv2D(filters=64, kernel_size=(5,5), strides=(1,1), padding='same', name='conv_3')) 
        self.add(Activation(activation='relu', name='relu_3'))
        self.add(BatchNormalization(name='batch_norm_3'))
        self.add(MaxPooling2D(pool_size=(2,2), padding='valid', name='max_pool_3')) # Pooling->28

        # Conv layer 4
        self.add(Conv2D(filters=32, kernel_size=(5,5), strides=(1,1), padding='same', name='conv_4')) 
        self.add(Activation(activation='relu', name='relu_4'))
        self.add(BatchNormalization(name='batch_norm_4'))

        # Fully Connected layer
        self.add(Flatten(name='flatten'))
        self.add(Dense(units=4096, activation='relu', name='dense_1'))
        self.add(Dense(units=2048, activation='relu', name='dense_2'))
        self.add(Dense(units=self.image_generator.num_classes, activation='softmax', name='output_layer'))

        self.compile(Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8), loss='sparse_categorical_crossentropy', metrics=[tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=self.image_generator.num_classes, average='micro', threshold=0.6)])

        if not k_fold:
            return xtrain, xtest, ytrain, ytest  
        else:
            return xset, yset


    def model_diagram(self):
        plt.figure(figsize=(10,10))
        plt.title("Epilypsis CNN network")
        plt.xticks([]), plt.yticks([])

        tf.keras.utils.plot_model(model=self, to_file=os.path.join('','EpilypsisSeqNet_schema.png'), show_shapes=True, show_layer_names=True)
        plt.imshow(imread(os.path.join('','EpilypsisSeqNet_schema.png')))
        plt.show()
        
    def clear_session(self):
        ops.reset_default_graph()
        tf.keras.backend.clear_session()
    
    def display_image_predictions(self):     
        xtrain, xtest, ytrain, ytest = self.load(k_fold=False)
        self.fit(
            xtrain,
            ytrain, 
            callbacks=EpilepsySeqNet.callbacks,
            batch_size=ImageGenerator.batch_size,
            epochs=50, 
            steps_per_epoch=xtrain.shape[0]//ImageGenerator.batch_size,
            verbose=True
        )
        ytest = np.array(ytest)
        yprob = np.array(self.predict(xtest))
        ypreds = yprob.argmax(axis=1)

        randomly_selected_samples = []
        random.seed(1234)
        for _ in range(20):
            random_i=random.randint(0,xtest.shape[0]-1)
            while random_i in randomly_selected_samples:
                random_i=random.randint(0,xtest.shape[0]-1)
        
        random.shuffle(randomly_selected_samples)
            
        random.shuffle(randomly_selected_samples)
        fig = plt.figure(figsize=(14,10))
        stop_signal = False
        row_samples = 4
        column_samples = 5
        gs = GridSpec(row_samples, column_samples)
        for i in range(row_samples):
            if stop_signal: break
            for j in range(column_samples):
                pos=column_samples*i+j
                if pos >= len(randomly_selected_samples): 
                    stop_signal = True
                    break
                x = float(yprob[randomly_selected_samples[pos]][ytest[randomly_selected_samples[pos]]])
                x*=100
                ax = fig.add_subplot(gs[i,j])
                ax.set_title(
                    f'True: {self.image_generator.class_names[ytest[randomly_selected_samples[pos]]]}\nPredicted: {self.image_generator.class_names[ypreds[randomly_selected_samples[pos]]]}, {"%.1f"%x}%', 
                    fontname="Times New Roman",
                    fontweight="bold",
                    fontsize=12,
                    loc='left'
                )
                img = np.array(xtest[randomly_selected_samples[pos]])
                img = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 255, 0)) if ypreds[randomly_selected_samples[pos]]==ytest[randomly_selected_samples[pos]] else cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(255, 0, 0))
                ax.imshow(tf.cast(img/255, tf.float32))
                ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(os.path.join('','project_results','figures','inference',f'prediction_labels_{len(randomly_selected_samples)}_samples.png'), dpi=300)
        plt.show()


    def __str__(self):
        return str(self.summary())

def predict_with_train_test_split():
    model=EpilepsySeqNet()
    xtrain, xtest, ytrain, ytest = model.load(k_fold=False)
    
    model.fit(
        xtrain,
        ytrain,
        batch_size = ImageGenerator.batch_size,
        epochs = 30,
        callbacks = EpilepsySeqNet.callbacks,
        steps_per_epoch = xtrain.shape[0]//ImageGenerator.batch_size,
        verbose = True
    )

    y_probs = np.array(model.predict(xtest))
    y_preds = y_probs.argmax(axis=1)
    for i in range(ytest.shape[0]):
        print(f'True:{model.image_generator.class_names[ytest[i]]}\tPredicted:{model.image_generator.class_names[y_preds[i]]}')
    
    print('\n\n',classification_report(ytest,y_preds,target_names=model.image_generator.class_names))


def scenario1():
    model = EpilepsySeqNet()
    xset, yset = model.load(k_fold=True)

    print(model.summary())
    xset, yset = np.array(xset), np.array(yset)

    cv_model = StratifiedKFold(n_splits=10, random_state=1234, shuffle=True)
    fold_no = 1
    rows=[]
    headers=['FOLD_NO','ACCURACY','RECALL','PRECISION','F1_SCORE']
    for train_indeces, test_indeces in cv_model.split(xset,yset):
        xtrain, xtest, ytrain, ytest = xset[train_indeces], xset[test_indeces], yset[train_indeces], yset[test_indeces]
        
        model.fit(
            xtrain,
            ytrain,
            epochs=30,
            callbacks=EpilepsySeqNet.callbacks,
            steps_per_epoch = xtrain.shape[0]//ImageGenerator.batch_size
        )
        
        rows.append([fold_no,accuracy_score(ytest,ypreds),recall_score(ytest,ypreds,average='macro'),precision_score(ytest,ypreds,average='macro'),f1_score(ytest,ypreds,average='macro')])
        fold_no+=1
        ypreds = np.array(model.predict(xtest)).argmax(axis=1)
        print(classification_report(ytest, ypreds, target_names=model.image_generator.class_names), end='\n\n')

    
        model.clear_session() # GPU clearence

    writer=pd.ExcelWriter(path=os.path.join('','stats','10-fold-cross_validation.xlsx'),mode='w')
    pd.DataFrame(rows,headers=headers).to_excel(writer,index=False)
    writer.close()


def scenario2():
    data=pd.read_excel(os.path.join('','stats','10-fold-cross_validation.xlsx'))
    plt.figure(figsize=(14,10))
    data_cls=data.columns.drop(labels=['FOLD_NO']).to_list()
    plt.title('CNN-Evaluation metrics based on 10-fold cross validation')
    for i,column in enumerate(data_cls):
        plt.boxplot(data[column],positions=[i],showmeans=True,meanprops=dict(marker='o',markerfacecolor='red',markeredgecolor='black'), capprops=dict(linewidth=2,linestyle='-'), whiskerprops=dict(marker='+',markeredgecolor='black'),flierprops=dict(marker='+',markeredgecolor='black'))
    plt.xticks(np.arange(len(data_cls)),data_cls)
    plt.show()

def scenario3():
    resnet_model=EpilepsySeqNet()
    resnet_model.display_image_predictions()





if __name__ == '__main__':
    # predict_with_train_test_split() # Sample predictions
    # scenario1()
    # scenario2()
    scenario3()

