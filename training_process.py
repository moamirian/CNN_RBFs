from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition
from sklearn.preprocessing import OneHotEncoder

from keras import backend as K
import keras
import keras_metrics
from random import shuffle
from keras import optimizers
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Conv2DTranspose, Flatten, Add, Activation, UpSampling2D, \
    Dropout, BatchNormalization, GlobalAveragePooling2D, Layer, Lambda
from keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from utils import *


def define_base_model(image_size):
    model = Sequential()
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=image_size, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=image_size, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', input_shape=image_size, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', input_shape=image_size, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    return model


def define_model(nClasses=2, input_height=224, input_width=224):
    # base_model = ResNet50(input_shape=(input_height,input_width, 3), include_top=False, weights='imagenet')
    base_model = define_base_model([input_height, input_width, 3])
    x = base_model.output

    # x = BatchNormalization(trainable=True)(x)
    # x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    # and a logistic layer -- let's say we have 200 classes
    x = Dense(256, activation=None)(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(256, activation='relu')(x)
    # x = BatchNormalization(trainable=False)(x)
    # x = Dropout(0.2)(x)
    #
    # x = Dropout(0.5)(x)
    embeddings = x
    x = RBF(10)(x)
    activations = x
    # x = Dense(25)(x)
    # x = Activation('relu')(x)

    x = Dense(output_dim=nClasses)(x)
    # x = Dense(output_dim=nClasses)(x)
    # prediction = x
    # x = Lambda(lambda x: x * 10)(x)
    # x = Dense(nClasses)(x)
    # x = Activation('relu')(x)
    predictions = Activation('softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model, base_model.input, embeddings, activations, predictions


def compute_embeddings(tf_sess, img_input, pre_embeddings, x_train, batch_size, iter_num):
    length = pre_embeddings.shape.as_list()[-1]
    start, samples = 0, batch_size * iter_num
    np_embeddings = -np.ones([samples, length])  # , dtype=np.uint8)
    for i in range(iter_num):
        x = tf_sess.run(x_train)
        embeds = tf_sess.run(pre_embeddings, feed_dict={img_input: x})
        np_embeddings[start:start + batch_size, :] = embeds
        start += batch_size
        print('Extract embeddings: {:05d}/{:05d}'.format(start, samples))
    print('Sum of the unfilled elements: {:d}'.format(np.sum(np_embeddings == -1)))
    return np_embeddings


def main():
    plt.close('all')
    digits = datasets.load_digits(n_class=10)
    X = digits.data
    y = digits.target

    print("Computing t-SNE embedding")
    t0 = time()
    X_tsne = manifold.TSNE(n_components=2, init='pca').fit_transform(X)
    # plot_MNIST(X_tsne, digits, "t-SNE embedding of the digits (time %.2fs)" % (time() - t0))
    plt.show()

    # Hyper-parameters
    batch_size = 256
    learning_rate = 0.001
    image_size = [28, 28]
    image_original = [28, 28]
    number_of_epoches = int(50)
    log_dir = "classification_mnist"
    number_of_classes = int(10)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf_sess = tf.Session(config=config)
    K.set_session(tf_sess)

    # Prepare the data
    train_data, test_data = keras.datasets.mnist.load_data()
    N = train_data[0].shape[0]
    M = test_data[0].shape[0]
    x_train, y_train = array_to_tensors(train_data, image_size, batch_size)
    x_test, y_test = array_to_tensors(test_data, image_size, batch_size)
    # x_train, y_train, N, x_valid, y_valid, M = dataProvider(data_dir, "train/Caucasian", True)

    # Define the model
    model, img_input, embeddings, activations, predictions = define_model(number_of_classes, image_size[0],
                                                                          image_size[1])
    model.summary()

    # Training
    opt = tf.contrib.opt.AdamWOptimizer(weight_decay=1e-4, learning_rate=1e-3)

    def custom_loss(activations):
        # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
        def loss(y_true, y_pred):
            loss_classification = keras.losses.categorical_crossentropy(y_true, y_pred)
            y_true2 = K.cast(K.equal(activations, K.expand_dims(K.max(activations, 1), 1)), dtype='float32')
            loss_inter = K.mean(K.sum(y_true2 - y_true2 * activations, 1))
            return 2 * loss_inter + loss_classification

        return loss

    model.compile(loss=custom_loss(activations),
                  optimizer=opt, metrics=['accuracy'])

    model.fit(x_train, y_train, steps_per_epoch=1)
    np_embeddings = compute_embeddings(tf_sess, model.input, embeddings, x_train, batch_size, N // batch_size + 1)
    print('Compute the cluster centers using kmeans!')
    kmeans = KMeans(10).fit(np_embeddings)
    print('Finish Computing the cluster centers using kmeans!')
    np_centers = kmeans.cluster_centers_

    center, sigmas, widths = model.layers[-3].get_weights()
    model.layers[-3].set_weights([np_centers, sigmas, widths])
    best_validation_performance = 0
    training_loss, validation_loss, training_accuracy, validation_accuracy, top_accuracy = [], [], [], [], []
    # '''
    validation_accuracy = 0
    for i in range(90):

        # Validation
        print(i)
        valid_scores = model.evaluate(x_test, y_test, steps=M // batch_size + 1)
        validation_accuracy = valid_scores[1]
        # valid_scores = [valid_scores[0]]+[item/(M//batch_size+1) for item in valid_scores[1:]]
        print("Validation loss: {:0.4f} - accuracy: {:0.4f}.".format(valid_scores[0],
                                                                     valid_scores[1]))

        all_embeddings, all_labels = get_embeddings(tf_sess, img_input, embeddings, [x_train, y_train],
                                                    5)  # N // batch_size+1)
        all_activations, all_labels = get_embeddings(tf_sess, img_input, activations, [x_train, y_train], 5)
        print(np.mean(1 - all_embeddings[:, 0][np.where(np.argmax(all_labels, 1) == 0)]))
        plt.close('all')
        centers = model.layers[-3].get_weights()[0]
        draw_and_save_tsne(i, all_embeddings, all_labels, 100,
                           'Validation accuracy = {:0.4f}'.format(validation_accuracy))
        draw_and_save_tsne(1000 + i, all_embeddings, all_labels, 100, '')
        draw_and_save_tsne(2000 + i, all_activations, all_labels, 100, '')
        # Training
        if i % 2 == 0:
            # switch_mode(model, 'netwrok')
            # model.compile(loss=custom_loss(embeddings), optimizer=opt, metrics=['accuracy'])
            train_scores = model.fit(x_train, y_train, steps_per_epoch=N // batch_size // 50, epochs=1)
        else:
            # switch_mode(model, 'rbf')
            # model.compile(loss=custom_loss(embeddings), optimizer=opt, metrics=['accuracy'])
            train_scores = model.fit(x_train, y_train, steps_per_epoch=N // batch_size // 10, epochs=1)
        print("Training loss: {:0.4f}.".format(train_scores.history["loss"][0]))

        for item in range(10):
            plt.close('all')
            if item == 9:
                draw_and_save(item, i, all_activations, all_labels, 8,
                              'Validation accuracy = {:0.4f}'.format(validation_accuracy))
            else:
                draw_and_save(item, i, all_activations, all_labels, 8)

        # Train accuracy:
        train_scores = model.evaluate(x_train, y_train, steps=N // batch_size + 1)
        print("Training loss: {:0.4f} - accuracy: {:0.4f}.".format(train_scores[0],
                                                                   train_scores[1]))
    '''
    for i in range(1):

        # Training
        train_scores = model.fit(x_train, y_train, steps_per_epoch=N // batch_size, epochs=1)
        print("Training loss: {:0.4f}.".format(train_scores.history["loss"][0]))

        # Train accuracy:
        train_scores = model.evaluate(x_train, y_train, steps=N // batch_size + 1)
        print("Training loss: {:0.4f} - accuracy: {:0.4f}.".format(train_scores[0],
                                                                   train_scores[1] ))

        # Validation
        valid_scores = model.evaluate(x_test, y_test, steps=M // batch_size + 1)
        validation_accuracy = valid_scores[1]
        #valid_scores = [valid_scores[0]]+[item/(M//batch_size+1) for item in valid_scores[1:]]
        print("Validation loss: {:0.4f} - accuracy: {:0.4f}.".format(valid_scores[0], valid_scores[1] ))

    all_activations, all_labels = get_embeddings(tf_sess, img_input, activations, [x_train, y_train], 5)
    test_images, test_labels, test_activations = get_embeddings(tf_sess, img_input, activations, [x_test, y_test], 1, True)  # N // batch_size+1)
    test_images, test_labels, test_activations  = test_images[:5, :, :, :], test_labels[:5, :], test_activations[:5, :]
    #test_images_random = np.random.rand(5, 28, 28, 3)
    #test_embeddings_random  = tf_sess.run(embeddings, feed_dict={img_input: test_images_random})
    #test_images = np.concatenate((test_images, test_images_random), 0)
    #test_embeddings = np.concatenate((test_embeddings, test_embeddings_random), 0)
    print(np.mean(1 - all_activations[:, 0][np.where(np.argmax(all_labels, 1) == 0)]))
    plt.close('all')
    for i in range(10):
        draw_and_save_test(i, all_activations, all_labels, test_activations, test_images, 8, title=None)
    '''
    '''
    for i in range(10):

        # Training
        train_scores = model.fit(x_train, y_train, steps_per_epoch=N // batch_size, epochs=1)
        print("Training loss: {:0.4f}.".format(train_scores.history["loss"][0]))

        # Train accuracy:
        train_scores = model.evaluate(x_train, y_train, steps=N // batch_size + 1)
        print("Training loss: {:0.4f} - accuracy: {:0.4f}.".format(train_scores[0],
                                                                   train_scores[1]))

        # Validation
        valid_scores = model.evaluate(x_test, y_test, steps=M // batch_size + 1)
        validation_accuracy = valid_scores[1]
        valid_scores = [valid_scores[0]]+[item/(M//batch_size+1) for item in valid_scores[1:]]
        print("Validation loss: {:0.4f} - accuracy: {:0.4f}.".format(valid_scores[0], valid_scores[1] ))

    all_embeddings, all_labels = get_embeddings(tf_sess, img_input, embeddings, [x_train, y_train], 5)
    test_images = np.random.rand(5, 28, 28, 3)
    test_embeddings  = tf_sess.run(embeddings, feed_dict={img_input: test_images})
    print(np.mean(1 - all_embeddings[:, 0][np.where(np.argmax(all_labels, 1) == 0)]))
    plt.close('all')
    centers = model.layers[-3].get_weights()[0]
    for i in range(10):
        draw_and_save_test(i, all_embeddings, all_labels, test_embeddings, test_images, 8, title=None)
    '''


if __name__ == '__main__':
    main()
