#python run_experiment.py --weights=imagenet --dataset=cifar10 --backbone=EfficientNet --augmentation=autogment --learning_rate=0.00002355 --weight_decay=1.090e-7 --rbf_dims=64 --batch_size=32 --loss_constant=0.1141 --centers=20 --dropout=0.0
#python run_experiment.py --weights=imagenet --dataset=cifar100 --backbone=EfficientNet --augmentation=autogment --learning_rate=0.0001873 --weight_decay=5.369e-7 --rbf_dims=32 --batch_size=64 --loss_constant=0.8557 --centers=50 --dropout=0.0
#python run_experiment.py --weights=imagenet --dataset=Oxford_IIIT_Pet --backbone=EfficientNet --augmentation=autogment --learning_rate=0.00007487 --weight_decay=1.150e-7 --rbf_dims=64 --batch_size=16 --loss_constant=1.067 --centers=50 --dropout=0.0
#python run_experiment.py --weights=imagenet --dataset=flowers --backbone=EfficientNet --augmentation=autogment --learning_rate=0.0001076 --weight_decay=0.000003843 --rbf_dims=16 --batch_size=8 --loss_constant=1.562 --centers=100 --dropout=0.0
#python run_experiment.py --weights=imagenet --dataset=fgvc-aircraft-2013b --backbone=EfficientNet --augmentation=autogment --learning_rate=0.0001103 --weight_decay=0.000001222 --rbf_dims=8 --batch_size=8 --loss_constant=0.5471 --centers=50 --dropout=0.0
#python run_experiment.py --weights=imagenet --dataset=CUB_200_2011 --backbone=EfficientNet --augmentation=autogment --learning_rate=0.0002603 --weight_decay=1.416e-8 --rbf_dims=32 --batch_size=32 --loss_constant=0.5156 --centers=50 --dropout=0.0

#python run_experiment.py --weights=imagenet --dataset=Oxford_IIIT_Pet --backbone=EfficientNet --augmentation=autogment --learning_rate=0.00007487 --weight_decay=1.150e-7 --rbf_dims=64 --batch_size=16 --loss_constant=1.067 --centers=50 --dropout=0.0
#python run_experiment.py --weights=imagenet --dataset=Oxford_IIIT_Pet --backbone=resnet --augmentation=None --learning_rate=0.00007699 --weight_decay=3.414e-7 --rbf_dims=32 --batch_size=32 --loss_constant=0.1846 --centers=20 --dropout=0.0
#python run_experiment.py --weights=imagenet --dataset=fgvc-aircraft-2013b --backbone=inception --augmentation=autogment --learning_rate=0.0001103 --weight_decay=0.000001222 --rbf_dims=8 --batch_size=8 --loss_constant=0.5470 --centers=50 --dropout=0.0

#pip install git+https://github.com/OverLordGoldDragon/keras-adamw.git
#pip install git+https://github.com/titu1994/keras-efficientnets.git
import tensorflow as tf
from time import time
import numpy as np
import os
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras_efficientnets import EfficientNetB0
from DatasetProvider import DatasetProvider
from keras.optimizers import *
from sklearn.cluster import KMeans
from shutil import copy
#from keras_adamw import AdamW, get_weight_decays, fill_dict_in_order

from keras import backend as K
import keras


from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Conv2DTranspose, Flatten, Add, Activation, UpSampling2D, \
    Dropout, BatchNormalization, GlobalAveragePooling2D, Layer, Lambda
from keras.models import Model
from utils import *
import pickle
import argparse
#import wandb

import sys
sys.path.append('./keras-auto-augment/')
from dataset import Cifar10ImageDataGenerator, SVHNImageDataGenerator, ImageNetImageDataGenerator

number_of_epoches = int(500)
parser = argparse.ArgumentParser()

parser.add_argument("--weight_decay", default=1e-6)
parser.add_argument("--learning_rate", default=1e-4)
parser.add_argument('--batch_size', default=64)
parser.add_argument('--loss_constant', default=1)
parser.add_argument('--rbf_dims', default=256)
parser.add_argument('--centers', default=100)
parser.add_argument('--dropout', default=0.5)
parser.add_argument('--backbone', default='efficientnet')
parser.add_argument('--weights', default='imagenet')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--augmentation', default='None')

args = parser.parse_args()
learning_rate = float(args.learning_rate)
weight_decay = float(args.weight_decay)
loss_constant = float(args.loss_constant)
batch_size = int(args.batch_size)
rbf_dims = int(args.rbf_dims)
number_of_centers = int(args.centers)
dropout = float(args.dropout)
backbone = str(args.backbone)
weights = str(args.weights)
dataset = str(args.dataset)
augmentation = str(args.augmentation)

dataset_dir = '../../datasets'
#wandb.init(entity="momi", project=dataset)

class ModelSetup():

    def __init__(self, image_size_dataset, backbone, rbf_dims, centers, dropout, nClasses, weights, **kwargs):
        self.weights = weights
        self.backbone = backbone
        self.rbf_dims = rbf_dims
        self.centers = centers
        self.dropout = dropout
        self.nClasses = nClasses
        self.image_size_dataset = image_size_dataset
        self.find_model_image_size()
        self.define_base_model()
        super(ModelSetup, self).__init__(**kwargs)

    def define_base_model(self):
        if 'efficientnet' in self.backbone.lower():
            self.base_model = EfficientNetB0(self.image_size_model, include_top=False, weights=self.weights)
        elif 'mobilenet' in self.backbone.lower():
            self.base_model = MobileNetV2(input_shape=self.image_size_model, include_top=False, weights=self.weights)
        elif 'resnet' in self.backbone.lower():
            self.base_model = ResNet50(input_shape=self.image_size_model, include_top=False, weights=self.weights)
        elif 'vgg' in self.backbone.lower():
            self.base_model = VGG16(input_shape=self.image_size_model, include_top=False, weights=self.weights)
        elif 'inception' in self.backbone.lower():
            self.base_model = InceptionV3(input_shape=self.image_size_model, include_top=False, weights=self.weights)

    def find_model_image_size(self):
        if 'inception' in self.backbone.lower():
            self.image_size_model = (229, 229, 3)
        else:
            self.image_size_model = (224, 224, 3)

    def define_model(self):

        # Define preprocessing:
        model_input = Input(self.image_size_dataset)
        x = Lambda(lambda x: tf.image.resize_images(x, self.image_size_model[:2]))(model_input)

        # Define the backbone model
        x = self.base_model(x)

        x = GlobalAveragePooling2D()(x)
        x = Dropout(dropout)(x)
        #x = BatchNormalization(trainable=True)(x)
        # x = Flatten()(x)
        x = Dense(self.rbf_dims)(x)
        #x = Dropout(0.5)(x)
        #x = Dense(512)(x)
        pre_embeddings = x
        RBF_layer = RBF(self.centers, self.nClasses)
        x = RBF_layer(x)
        #x = Dropout(0.5)(x)
        #embeddings = x
        embeddings = RBF_layer.embeddings
        #predictions = Dense(output_dim=self.nClasses, activation='softmax')(x)
        predictions = Activation('softmax')(x)
        #model = Model(inputs=base_model.input, outputs=predictions)
        model = Model(inputs=model_input, outputs=predictions)

        return model, model.input, pre_embeddings, embeddings, predictions
    
def compute_embeddings(tf_sess, img_input, pre_embeddings, x_train):
    length = pre_embeddings.shape.as_list()[-1]
    start, samples = 0, x_train.shape[0]
    np_embeddings = -np.ones([samples, length])#, dtype=np.uint8)
    while start < samples:
        embeds = tf_sess.run(pre_embeddings, feed_dict={img_input: x_train[start:start+batch_size, :, :, :]})
        np_embeddings[start:start+batch_size, :] = embeds
        start += batch_size
        if np.mod(start, (samples//10//batch_size)*batch_size) == 0:
            print('Compute embeddings: {:05d}/{:05d}'.format(start, samples))
    #print('Sum of the unfilled elements: {:d}'.format(np.sum(np_embeddings==-1)))
    return np_embeddings

def augmentation_prepration(args, auto_agment=True):
    args.depth = int(28)
    args.width = int(10)
    args.epochs = int(number_of_epoches)
    args.cutout = False
    args.auto_augment = auto_agment
    return args

def save_dict(dict, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(dict, f, pickle.HIGHEST_PROTOCOL)

def main():
   
    # Dictionary of input parameters
    parameters_dict = {'backbone': backbone, 'rbf_dims': rbf_dims, 'centers': number_of_centers, 'learning_rate': learning_rate,
                       'weight_decay': weight_decay,'loss_constant': loss_constant, 'batch_size': batch_size,
                       'dropout': dropout, 'weights': weights, 'dataset': dataset, 'augmentation': augmentation}

    # Make directory:
    list_of_parameters = ['backbone', 'weights', 'augmentation', 'rbf_dims', 'centers', 'learning_rate']
    model_dir, list_of_params = '', ['{}_{}_'.format(key, value) for key, value in parameters_dict.items() if key in list_of_parameters]
    for item in list_of_params:
        model_dir += item
    model_dir = os.path.join('models', dataset, model_dir)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    copy('run_experiment.py', os.path.join(model_dir, 'run_experiment.py'))
    save_dict(parameters_dict, os.path.join(model_dir, 'parameters'))

    # Set the keras session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=config)
    K.set_session(tf_sess)

    # Prepare the data
    data_provider = DatasetProvider(dataset_dir, dataset)
    x_train, y_train, x_test, y_test = data_provider.load_data()
    number_of_classes, image_size_dataset = data_provider.number_of_classes, data_provider.image_size

    # AutoAugment:
    args_auto = augmentation_prepration(args)
    if 'imagenet' in augmentation.lower():
        datagen = ImageNetImageDataGenerator(args_auto)
    elif 'cifar' in augmentation.lower():
        datagen = Cifar10ImageDataGenerator(args_auto)
    elif 'svhn' in augmentation.lower():
        datagen = SVHNImageDataGenerator(args_auto)
    else:
        datagen = Cifar10ImageDataGenerator(args_auto)

    x_test = datagen.standardize(x_test)
    x_train2 = datagen.standardize(x_train)

    # Define the model
    model_definer = ModelSetup(x_train.shape[1:], backbone, rbf_dims, number_of_centers, dropout, number_of_classes, weights)
    model, img_input, pre_embeddings, embeddings, predictions = model_definer.define_model()
    model.summary()
    #'''
    idx = -2
    centers = model.layers[idx].weights[0]
    centers_dist = pairwise_dist(centers)

    # Define loss:
    def custom_loss(embeddings):
        # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
        def loss(y_true, y_pred):
            loss_classification = keras.losses.categorical_crossentropy(y_true, y_pred)
            y_true2 = K.cast(K.equal(embeddings, K.expand_dims(K.max(embeddings, 1), 1)), dtype='float32')
            loss_inter = K.mean(K.sum(y_true2 - y_true2 * embeddings, 1))            
            return loss_constant*loss_inter + loss_classification
        return loss

    # Training
    opt = tf.contrib.opt.AdamWOptimizer(weight_decay=weight_decay, learning_rate=learning_rate)
    #wd_dict = get_weight_decays(model)
    #weight_decay_dict = fill_dict_in_order(wd_dict,[weight_decay for itm in wd_dict.keys()])
    #opt = AdamW(lr=learning_rate, weight_decays=weight_decay_dict, use_cosine_annealing=True, total_iterations=number_of_epoches) 
    #opt = SGD(lr=learning_rate, momentum=weight_decay)

    model.compile(loss=custom_loss(embeddings), optimizer=opt, metrics=['accuracy'])
    #model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    # Initialize the centers:
    idx = -2
    if augmentation=='None':
        model.fit(x_train2, y_train, batch_size=batch_size, epochs=1)        
    else:
        model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=1)
    np_embeddings = compute_embeddings(tf_sess, img_input, pre_embeddings, x_train2)
    print('Compute the cluster centers using kmeans!')
    kmeans = KMeans(number_of_centers).fit(np_embeddings)
    print('Finish Computing the cluster centers using kmeans!')    
    np_centers = kmeans.cluster_centers_

    sigmas = model.layers[idx].weights[1]
    shifts = model.layers[idx].weights[2]
    all_weights = model.layers[idx].get_weights()
    all_weights[0] = np_centers
    #model.layers[idx].set_weights([np_centers, np.ones(sigmas.shape.as_list()), np.ones(shifts.shape.as_list())]) 
    model.layers[idx].set_weights(all_weights)         
    best_validation_performance = 0
    training_loss, validation_loss, training_accuracy, validation_accuracy, top_accuracy = [], [], [], [], []
    for i in range(number_of_epoches):
        # Training and evaluation of training set
        #'''
        if augmentation=='None':
            model.fit(x_train2, y_train, batch_size=batch_size)			
        else:
            model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) // batch_size,
                            validation_data=(x_test, y_test), epochs=1)
        train_predictions = model.evaluate(x_train2, y_train, batch_size=batch_size)
        #train_scores = model.fit(x_train, y_train, steps_per_epoch=N//batch_size+1)
        #train_predictions = model.evaluate(x_train, y_train, steps=N//batch_size+1)        
        print("Training loss: {:0.4f}, Training accuracy: {:0.4f}.".format(*train_predictions))
        #'''
        # Evaluation on validation set:
        test_predictions = model.evaluate(x_test, y_test, batch_size=batch_size)
       
        if test_predictions[1] > best_validation_performance:
            best_validation_performance = test_predictions[1]
            filename = os.path.join(model_dir, 'model_best_performance.h5')
            model.save_weights(filename)
            print("Saved model to disk")
        if i % 1 == 0:
            filename = os.path.join(model_dir, 'model_{:03d}_{:0.4f}.h5'.format(i, test_predictions[1]))
            model.save_weights(filename)
            '''
            # serialize model to JSON            
            filename_json = filename.replace('h5', 'json')
            model_json = model.to_json()
            with open(filename_json, "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            model.save_weights(filename)
            '''
            print("Saved model to disk")
        #test_predictions = model.evaluate(x_test, y_test, steps=M//batch_size+1)        
        #print("Validation loss: {:0.4f}, Validation accuracy: {:0.4f}, best performance: {:0.4f}.".format(*test_predictions, best_validation_performance))
        #wandb.log({'accuracy': best_validation_performance, 'train_loss': train_predictions[0], 'train_accuracy': train_predictions[1], 'validation_loss': test_predictions[0], 'validation_accuracy': test_predictions[1]}, step=i)
        training_loss.append(train_predictions[0])
        training_accuracy.append(train_predictions[1])
        validation_loss.append(test_predictions[0])
        validation_accuracy.append(test_predictions[1])
        top_accuracy.append(best_validation_performance)
        print("Best validation performance: {:0.4f}.".format(best_validation_performance))
        if train_predictions[1] > 0.99:
            break        
        #if i >= 10 and (best_validation_performance < 0.2 or top_accuracy[i-3] >= top_accuracy[i]):
        #    break
    print("Best validation performance: {:0.4f}.".format(best_validation_performance))
    np.save(os.path.join(model_dir, 'training_loss.npy'), np.array(training_loss))
    np.save(os.path.join(model_dir, 'training_accuracy.npy'), np.array(training_accuracy))
    np.save(os.path.join(model_dir, 'validation_loss.npy'), np.array(validation_loss))
    np.save(os.path.join(model_dir, 'validation_accuracy.npy'), np.array(validation_accuracy))
    np.save(os.path.join(model_dir, 'top_accuracy.npy'), np.array(top_accuracy))
    
if __name__ == '__main__':
    main()
