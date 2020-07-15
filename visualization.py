import tensorflow as tf
from time import time
import numpy as np
from random import shuffle
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
from matplotlib import pyplot as plt
from matplotlib import ticker, cm
#from keras_adamw import AdamW, get_weight_decays, fill_dict_in_order
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import sys
sys.path.append('../keras-auto-augment/')
from PIL import Image
from dataset import Cifar10ImageDataGenerator, SVHNImageDataGenerator, ImageNetImageDataGenerator
from keras import backend as K
import keras
import argparse
import glob
import pickle
import random
random.seed(10)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Conv2DTranspose, Flatten, Add, Activation, UpSampling2D, \
    Dropout, BatchNormalization, GlobalAveragePooling2D, Layer, Lambda
from keras.models import Model
from utils import *
import sys

# Model and dataset directory:
batch_size = 256
cluster_idx = 6
figure_size = (15, 15)
checkpoint_idx = -1
#distance_range = 2
dataset = 'Oxford_IIIT_Pet'
dataset_dir = '../../datasets'
results_dir = './visualization'
number_of_similar_samples = 15
number_of_effective_clusters = 5
number_of_samples_per_cluster = 30
model_folder = 'backbone_EfficientNet_rbf_dims_64_centers_50_learning_rate_7.487e-05_weights_imagenet_augmentation_autogment_'
model_dir = os.path.join('models', dataset, model_folder)

# Import the model setup function and read the hyperparameters:
sys.path.append(model_dir)
from run_experiment import ModelSetup

def compute_embeddings_activations(tf_sess, img_input, tensors, x_train):
    length = tensors[0].shape.as_list()[-1]
    length1 = tensors[1].shape.as_list()[-1]
    length2 = tensors[2].shape.as_list()[-1]
    start, samples = 0, x_train.shape[0]
    np_embeddings = -np.ones([samples, length])#, dtype=np.uint8)
    np_activations = -np.ones([samples, length1])
    np_predictions = -np.ones([samples, length2])
    while start < samples:
        embeds, acts, preds = tf_sess.run(tensors, feed_dict={img_input: x_train[start:start+batch_size, :, :, :]})
        np_embeddings[start:start+batch_size, :] = embeds
        np_activations[start:start+batch_size, :] = acts
        np_predictions[start:start+batch_size, :] = preds       
        start += batch_size
        print('Extract embeddings and predictions: {:05d}/{:05d}'.format(start, samples))
    print('Sum of the unfilled elements in embeddings: {:d}'.format(np.sum(np_embeddings==-1)))
    print('Sum of the unfilled elements in activations: {:d}'.format(np.sum(np_activations==-1)))
    print('Sum of the unfilled elements in predictions: {:d}'.format(np.sum(np_predictions==-1)))
    return np_embeddings, np_activations, np_predictions

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def augmentation_prepration():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.depth = int(28)
    args.width = int(10)
    args.epochs = int(1)
    args.cutout = False
    args.auto_augment = True
    return args

def plot_ground(w, distance_range=2):
    fig = plt.figure(figsize=figure_size)
    #ax = fig.add_subplot(111, aspect='equal')
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax = fig.add_axes([left, bottom, width, height])
    xlist = np.linspace(-distance_range, distance_range, 100)
    ylist = np.linspace(-distance_range, distance_range, 100)
    X, Y = np.meshgrid(xlist, ylist)
    Z = np.sqrt(X ** 2 + Y ** 2)
    cp = plt.contour(X, Y, Z, colors=6*['red'], linewidths=1.0)
    ax.clabel(cp, inline=True, fontsize=16)
    cp = plt.contourf(X, Y, 1-Z/w, cmap=cm.PuBu_r)

    plt.axis('equal')
    plt.axis('tight')
    #plt.colorbar()

    tick_values = 0.8*distance_range
    xy_labels = np.around(np.abs(np.linspace(-tick_values, tick_values, 5)), decimals=1)
    xy_ticks = np.linspace(-tick_values, tick_values, 5)
    plt.xticks(xy_ticks, xy_labels)
    plt.yticks(xy_ticks, xy_labels)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(24)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(24)
    #ax.set_title('Cluster')
    ax.set_xlabel('Distance to the cluster center', fontsize=24)
    ax.set_ylabel('Distance to the cluster center', fontsize=24)
    #plt.show()
    #plt.show()
    return ax, plt

def compute_metric_distance(a, b, r=None):
    # Compute the distances between a sample a and samples in matrix b based on the trained metric r
    a, b = np.squeeze(a), np.squeeze(b)

    if r is None:
        print('Define the distance based on dot product!')
        def compute_dist(_a, _b, _r):
            # Compute distance between two samples
            distance = np.dot(_a, _b)
            return distance
    else:
        print('Define the distance based on the trained metric!')
        r = np.squeeze(r)
        def compute_dist(_a, _b, _r):
            # Compute distance between two samples
            diff = np.expand_dims(_a - _b, 0)
            distance = np.matmul(np.matmul(diff, np.diag(_r)), np.transpose(diff))
            return distance

    if len(a.shape) == 1 and len(b.shape) == 1:
        distances = compute_dist(a, b, r)
    elif len(a.shape) == 1 and len(b.shape) != 1:
        distances = np.zeros(b.shape[0])
        for i in range(b.shape[0]):
            distances[i] = compute_dist(a, b[i, :], r)
    elif len(a.shape) != 1 and len(b.shape) == 1:
        distances = np.zeros(a.shape[0])
        for i in range(a.shape[0]):
            distances[i] = compute_dist(a[i, :], b, r)
    else:
        distances = np.zeros([a.shape[0], b.shape[0]])
        for i in range(a.shape[0]):
            for j in range(b.shape[0]):
                distances[i, j] = compute_dist(a[i, :], b[j, :], r)
    return np.squeeze(distances)

def find_samples(np_embeddings, width, distance_range=2):
    number_of_bins = 15
    samples = [1, 0, 0, 3, 0, 5, 0, 9, 0, 12, 0, 0, 16, 0, 0, 0]
    distances = np.sqrt(width*(1-np_embeddings)) 
    #if distance_range < np.min(distances):
    #    distance_range = (np.min(distances)+np.max(distances))/2
    bin_edges  = np.linspace(0, distance_range, number_of_bins)
    #samples_per_bin, = int(number_of_samples/number_of_bins), []
    indecies, theta = [], []
    for i in range(len(bin_edges)-1):
        samples_per_bin = samples[i]
        if samples_per_bin > 0:
            found_indecies = list(np.where(np.bitwise_and(distances>bin_edges[i], distances<bin_edges[i+1]))[0])
            shuffle(found_indecies)
            found_indecies = found_indecies[:samples_per_bin]
            indecies += list(found_indecies)
            N = len(found_indecies)
            theta += list(np.linspace(0, 2*np.pi*(1-1/np.max([1, N])), N)+(np.pi/18)*(np.random.rand(N)-0.5))
            samples_per_bin = samples[i]            
    return np.array(indecies), np.array(theta)

def plot_images(ax, images, embeddings, w, theta, image_size=[64, 64]):
    distances = np.sqrt(w*(1-embeddings))
    for i in range(images.shape[0]):
        x, y = distances[i]*np.cos(theta[i]), distances[i]*np.sin(theta[i])
        im = np.array(Image.fromarray(images[i, :, :, :]).resize(image_size))
        imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(im), np.array([x, y]))
        ax.add_artist(imagebox)

def plot_single_images(ax, images, embeddings, w, theta, distance_range=2, image_size=[96, 96]):
    distances = np.sqrt(w*(1-embeddings))
    x, y = distances*np.cos(theta), distances*np.sin(theta)
    if x < -distance_range:
        x, y = distance_range, distance_range
    im = np.array(Image.fromarray(images[:, :, :]).resize(image_size))
    im = prepare_test_image(im)
    imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(im), np.array([x, y]))
    ax.add_artist(imagebox)

def discriminant_clusters(activatios, weights, clusters):
    values = np.multiply(activatios, weights)
    indecies = np.flip(np.argsort(values)[-clusters-1:])
    return indecies

def prepare_test_image(image):  
    image[0:5, :, :], image[-5:, :, :] = [255, 0, 0], [255, 0, 0]
    image[:, 0:5, :], image[:, -5:, :] = [255, 0, 0], [255, 0, 0]
    return image

def main():

    paramters_dict = load_obj(os.path.join(model_dir, 'parameters'))

    # Set the keras session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=config)
    K.set_session(tf_sess)

    # Prepare the data
    data_provider = DatasetProvider(dataset_dir, dataset)
    x_train_org, y_train, x_test_org, y_test = data_provider.load_data()
    number_of_classes, image_size_dataset = data_provider.number_of_classes, data_provider.image_size
    datagen = Cifar10ImageDataGenerator(augmentation_prepration())
    x_train, x_test = datagen.standardize(x_train_org), datagen.standardize(x_test_org)
    #x_train, x_test = x_train[:1000, :, :, :], x_test[:500, :, :, :]
    #y_train, y_test = y_train[:1000, :], y_test[:500, :]

    # Build the model:
    backbone, rbf_dims, number_of_centers = paramters_dict['backbone'], paramters_dict['rbf_dims'], paramters_dict['centers']
    dropout, weights = paramters_dict['dropout'], paramters_dict['weights']
    model_definer = ModelSetup(x_train.shape[1:], backbone, rbf_dims, number_of_centers, dropout, number_of_classes, weights)
    model, img_input, embeddings, activations, predictions = model_definer.define_model()
    model.summary()

    # Load the model:
    filenames = sorted(glob.glob(os.path.join(model_dir, '*.h5')))
    print(filenames)
    model.load_weights(filenames[checkpoint_idx])
    print(filenames[checkpoint_idx])
    cluster_coordinate = model.layers[-2].get_weights()[0]
    distance_metric = model.layers[-2].get_weights()[1]
    output_weights = model.layers[-2].get_weights()[-1][1:, :]
    width = model.layers[-2].get_weights()[2]
    np.set_printoptions(precision=4, suppress=True)

    # Compute the embeddings, activations and Test the accuracy:    
    embeddings_train, activations_train, train_predictions = compute_embeddings_activations(tf_sess, img_input, [embeddings, activations, model.output], x_train)
    embeddings_test, activations_test, test_predictions = compute_embeddings_activations(tf_sess, img_input, [embeddings, activations, model.output], x_test)
    train_accuracy = np.mean(np.argmax(train_predictions, 1)==np.argmax(y_train, 1))
    test_accuracy = np.mean(np.argmax(test_predictions, 1)==np.argmax(y_test, 1))
    print('Testing the loaded model. Training accuracy: {:0.4f}, test accuracy: {:0.4f}.'.format(train_accuracy, test_accuracy)) 

    # Plot the contour of the distances to the center of the classifier (background color is proportional to the activation value)
    _activations = activations_train[:, cluster_idx]
    dists = np.sqrt(width[0, cluster_idx]*(1-_activations))
    min_dist, max_dist = np.min(dists), np.max(dists)
    distance_range = np.max([2, (min_dist+max_dist)/2])
    ax, fig = plot_ground(width[0, cluster_idx], distance_range)

    # For a given cluster get the sample with lower metric distance than 4/5 (histogram)
    samples_indices, theta = find_samples(_activations, width[0, cluster_idx], distance_range)

    # Make the scatter plot to show the images based on their distance and a random angle to the center
    plot_images(ax, x_train_org[samples_indices, :, :, :], activations_train[samples_indices, cluster_idx], width[0, cluster_idx], theta)

    # Save the complete image
    fig.savefig('sample_cluster.png', bbox_inches = 'tight')
    random.seed(10)
    rand_idx = [np.random.randint(0, x_test.shape[0]) for x in range(20)]
    rand_idx = [779, 56, 1665, 3526, 2523, 435, 2701, 2428, 3429, 952]    
    for sp in range(len(rand_idx)):
        # Test sample ID:
        test_idx = rand_idx[sp]

        # Find the similar samples to a given test image in training samples based on embedding (trained metric distance)
        sample_label = np.argmax(y_test[test_idx, :])
        sample_prediction = np.argmax(test_predictions[test_idx, :])
        #distances = cosine_similarity(embeddings_test, embeddings_train)[test_idx, :] 
        #distances = euclidean_distances(embeddings_test, embeddings_train)[test_idx, :]
        distances = compute_metric_distance(embeddings_test[test_idx, :], embeddings_train, distance_metric)        
        idx = np.argsort(distances)[:number_of_similar_samples]
        idx2 = np.argsort(distances)[-number_of_similar_samples:]
        save_dir = os.path.join(results_dir, '{:02d}_{}'.format(test_idx, str(sample_label==sample_prediction)))
        print(save_dir)
        if not os.path.isdir(save_dir):
            os.makedirs(os.path.join(save_dir, 'embeddings'))
            os.makedirs(os.path.join(save_dir, 'embeddings_dissimilar'))
            os.makedirs(os.path.join(save_dir, 'activations'))
            os.makedirs(os.path.join(save_dir, 'clusters_correct'))
            os.makedirs(os.path.join(save_dir, 'clusters_wrong'))
        Image.fromarray(x_test_org[test_idx]).resize([256, 256]).save(os.path.join(save_dir, 'test_image.png'))
        for i in range(len(idx)):
            Image.fromarray(x_train_org[idx[i]]).resize([256, 256]).save(os.path.join(save_dir, 'embeddings', '{:02d}.png'.format(i)))
            #Image.fromarray(x_train_org[idx2[i]]).resize([256, 256]).save(os.path.join(save_dir, 'embeddings_dissimilar', '{:02d}.png'.format(i)))

        '''
        # Find the most influential clusters for the correct class
        cluster_indecies = discriminant_clusters(activations_test[test_idx, :], output_weights[:, sample_label], number_of_effective_clusters)
        test_image_prepared = x_test_org[int(test_idx)]#prepare_test_image(x_test_org[int(test_idx), :, :, :])

        
        # Find the most influential clusters for the highest scoring wrong class
        sample_predictions = test_predictions[test_idx, :]
        sample_predictions[sample_label] = 0
        wrong_class_idx = np.argmax(sample_predictions)
        cluster_indecies_wrong = discriminant_clusters(activations_test[test_idx, :], output_weights[:, wrong_class_idx], number_of_effective_clusters)
        distance_range_sample = 2*np.max(np.array([np.sqrt(width[0, i]*(1-activations_test[test_idx, i])) for i in list(cluster_indecies)+list(cluster_indecies_wrong)]))
        
        # Draw the most influential clusters for the correct class and position the test sample within the clusters
        for center_idx in cluster_indecies:
            plt.close('all')
            ax, fig = plot_ground(width[0, center_idx], distance_range_sample)
            samples_indices, theta = find_samples(activations_train[:, center_idx], width[0, center_idx], distance_range_sample)
            if len(samples_indices) > 0:
                plot_images(ax, x_train_org[samples_indices, :, :, :], activations_train[samples_indices, center_idx], width[0, center_idx], theta)	
                plot_single_images(ax, test_image_prepared, activations_test[test_idx, center_idx], width[0, center_idx], np.pi/2, distance_range_sample)
                fig.savefig(os.path.join(save_dir, 'clusters_correct', '{:02d}.png'.format(center_idx)), bbox_inches = 'tight')
            else:
                print('Empty cluster: {:d}!'.format(center_idx))

        # Draw the most influential clusters for the highest scoring wrong class and position the test sample within the clusters
        for center_idx in cluster_indecies_wrong:
            plt.close('all')
            ax, fig = plot_ground(width[0, center_idx], distance_range_sample)
            samples_indices, theta = find_samples(activations_train[:, center_idx], width[0, center_idx], distance_range_sample)
            if len(samples_indices) > 0:            
                plot_images(ax, x_train_org[samples_indices, :, :, :], activations_train[samples_indices, center_idx], width[0, center_idx], theta)
                plot_single_images(ax, test_image_prepared, activations_test[test_idx, center_idx], width[0, center_idx], -np.pi/2, distance_range_sample)
                fig.savefig(os.path.join(save_dir, 'clusters_wrong', '{:02d}.png'.format(center_idx)), bbox_inches = 'tight')
            else:
                print('Empty cluster: {:d}!'.format(center_idx))

        # Find the similar samples to a given test image in training sampels based on activations (do product)
        distances = compute_metric_distance(activations_test[test_idx, :], activations_train)
        idx = np.argsort(distances)[-number_of_similar_samples:]
        for i in range(len(idx)):
            Image.fromarray(x_train_org[idx[i]]).resize([256, 256]).save(
                os.path.join(save_dir, 'activations', '{:02d}.png'.format(i)))
        '''
if __name__ == '__main__':
    main()
