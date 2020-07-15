import keras as k
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import manifold, datasets, decomposition
from keras.layers import Dense, Conv2D, MaxPooling2D, Input, Conv2DTranspose, Flatten, Add, Activation, UpSampling2D, Dropout, BatchNormalization, GlobalAveragePooling2D, Layer, Lambda

def plot_MNIST(X, y, scale, title=None, draw_circle=True):
    #x_min, x_max = np.min(X, 0), np.max(X, 0)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    r = 1/scale
    X /= 2*scale
    X += 0.5
    if draw_circle:
        theta = np.linspace(0, 2*np.pi, 1000)
        _x = r * np.cos(theta)
        _y = r * np.sin(theta)
        plt.plot(_x + 0.5, _y + 0.5, 'g', linewidth=2.5)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 16})
    '''
    if hasattr(offsetbox, 'AnnotationBbox'):
        ## only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(digits.data.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 5e-3:
                ## don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    '''
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], [1, 0.5, 0, 0.5, 1]), plt.yticks([0, 0.25, 0.5, 0.75, 1.0], [1, 0.5, 0, 0.5, 1])
    #plt.xticks([0, 0.25, 0.5, 0.75, 1.0], [scale, int(scale/2), 0.5, int(scale/2), scale]), plt.yticks([1, 0.5, 0, 0.5, 1], [scale, int(scale/2), 0.5, int(scale/2), scale])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(24)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(24)
    if title is not None:
        plt.title(title)

def plot_MNIST2(X, y, test_samples, scale, title=None, draw_circle=True):
    #x_min, x_max = np.min(X, 0), np.max(X, 0)
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    r = 1/scale
    X /= 2*scale
    test_samples[0] /= 2*scale
    X += 0.5
    test_samples[0] += 0.5
    if draw_circle:
        theta = np.linspace(0, 2*np.pi, 1000)
        _x = r * np.cos(theta)
        _y = r * np.sin(theta)
        plt.plot(_x + 0.5, _y + 0.5, 'g', linewidth=2.5)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 16})
    if hasattr(offsetbox, 'AnnotationBbox'):
        ## only print thumbnails with matplotlib > 1.0
        for i in range(test_samples[0].shape[0]):
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(test_samples[1][i, :, :, :], cmap=plt.cm.gray_r),
                                                test_samples[0][i, :])
            ax.add_artist(imagebox)
    plt.axis([0, 1, 0, 1])
    plt.grid()
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], [scale, int(scale/2), 0.5, int(scale/2), scale]), plt.yticks([1, 0.5, 0, 0.5, 1],
                                                                        [scale, int(scale/2), 0.5, int(scale/2), scale])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(24)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(24)
    if title is not None:
        plt.title(title)

'''
def plot_MNIST2(ax, X, digits, test_samples, title=None):
    x_min, x_max = np.min([np.min(X, 0), np.min(test_samples[0], 0)]), np.max(
        [np.max(X, 0), np.max(test_samples[0], 0)])
    y = digits.target
    X = (X - x_min) / (x_max - x_min)  # scale the values to fit
    test_samples[0] = (test_samples[0] - x_min) / (x_max - x_min)
    # ax.figsize((10, 10))
    # ax = plt.subplot(111)
    for i in range(X.shape[0]):
        ax.text(X[i, 0], X[i, 1], str(digits.target[i]),
                color=plt.cm.Set1(y[i] / 10.),
                fontdict={'weight': 'bold', 'size': 9})
    if hasattr(offsetbox, 'AnnotationBbox'):
        ## only print thumbnails with matplotlib > 1.0
        for i in range(test_samples[0].shape[0]):
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(test_samples[1][i, :, :, :], cmap=plt.cm.gray_r),
                                                test_samples[0][i, :])
            ax.add_artist(imagebox)
    ax.set_aspect('equal')
    plt.sca(ax)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
'''

def pairwise_dist(c):
    sigma = tf.ones([1, c.get_shape().as_list()[1]])
    # c = tf.slice(self.center, [0, 0], [1, -1])
    # c = tf.nn.relu(c)
    output_dim = c.get_shape().as_list()[0]
    center = c
    A = c
    c = tf.slice(center, [0, 0], [1, -1])
    s = tf.slice(sigma, [0, 0], [1, 1])
    A_c = tf.subtract(A, c)
    dists = tf.expand_dims(tf.diag_part(tf.matmul(tf.multiply(A_c, s), tf.transpose(A_c))), -1)
    for i in range(output_dim - 1):
        c = tf.slice(center, [i + 1, 0], [1, -1])
        # c = tf.nn.relu(c)
        s = tf.slice(sigma, [0, i + 1], [1, 1])
        A_c = tf.subtract(A, c)
        temp = tf.expand_dims(tf.diag_part(tf.matmul(tf.multiply(A_c, s), tf.transpose(A_c))), -1)
        dists = tf.concat((dists, temp), -1)
    return dists


def compute_dists(all_embeddings, prototype_index):
    distances = (1 - all_embeddings[:, int(prototype_index)])
    distances += np.random.rand(distances.size) / 10
    cut_constant = 8
    distances[distances > cut_constant] = cut_constant
    distances /= cut_constant
    theta = np.random.rand(distances.size) * 2 * np.pi
    X_dist = np.transpose([distances * np.cos(theta), distances * np.sin(theta)])
    return X_dist

class DenseSoftmax(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(DenseSoftmax, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(DenseSoftmax, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        Normalize = Lambda(lambda x: tf.nn.softmax(tf.nn.relu(tf.multiply(x, 10.0)), 0))
        return K.dot(x, Normalize(self.kernel))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

class RBF(Layer):

    def __init__(self, centers, output_dim, **kwargs):
        self.centers = centers		
        self.output_dim = output_dim
        super(RBF, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.center = self.add_weight(name='center',
                                      shape=(self.centers, input_shape[1]),
                                      initializer='uniform',
                                      trainable=True)
        self.sigma = self.add_weight(name='sigma',
                                      shape=(1, input_shape[1]),
                                      initializer=k.initializers.RandomUniform(minval=0.5, maxval=1.0),
                                      trainable=True)
        self.width = self.add_weight(name='width',
                                      shape=(1, self.centers),
                                      initializer='ones',
                                      trainable=True)
        self.output_weights = self.add_weight(name='output_weights',
                                      shape=(1+self.centers, self.output_dim),
                                      initializer=k.initializers.RandomUniform(minval=0.5, maxval=1.0),                                      
                                      trainable=True)
        self.apply = Lambda(lambda x: self.apply_rbf(x))
        super(RBF, self).build(input_shape)  # Be sure to call this at the end

    def pairwise_dist(self, A):
        #sigma = tf.nn.relu(self.sigma)
        c = tf.slice(self.center, [0, 0], [1, -1])
        #s = tf.slice(sigma, [0, 0], [1, 1])
        sigma = tf.nn.relu(self.sigma)
        s = tf.diag(tf.squeeze(sigma))
        A_c = tf.subtract(A, c)
        dists = tf.expand_dims(tf.diag_part(tf.matmul(tf.matmul(A_c, s), tf.transpose(A_c))), -1)
        for i in range(self.centers-1):
            c = tf.slice(self.center, [i+1, 0], [1, -1])
            A_c = tf.subtract(A, c)
            temp = tf.expand_dims(tf.diag_part(tf.matmul(tf.matmul(A_c, s), tf.transpose(A_c))), -1)
            dists = tf.concat((dists, temp), -1)
        return dists

    def apply_rbf(self, x):
        D = self.pairwise_dist(x)
        width = tf.nn.relu(self.width)
        D2 = tf.add(tf.multiply(-D, width), 1)
        self.embeddings = D2
        D2_bias = tf.concat([tf.ones([tf.shape(D2)[0], 1], tf.float32), D2], axis=1)
        output = tf.matmul(D2_bias, tf.nn.relu(self.output_weights))
        #output = tf.matmul(D2_bias, self.output_weights)        
        return output

    def call(self, x):
        return self.apply_rbf(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def define_base_model(image_size):
    model = Sequential()
    '''
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=image_size))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    '''
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=image_size, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=image_size, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', input_shape=image_size, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation=None, input_shape=image_size, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    return model


def define_base_model(image_size):
    model = Sequential()
    '''
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=image_size))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    '''
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=image_size, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', input_shape=image_size, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation=None, input_shape=image_size, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    return model


def define_model(nClasses=2, input_height=224, input_width=224):
    # base_model = ResNet50(input_shape=(input_height,input_width, 3), include_top=False, weights='imagenet')
    base_model = define_base_model(image_size + [3])
    x = base_model.output
    '''
    # add a global spatial average pooling layer
    x = Flatten()(x)
    x = Dense(128, activation=tf.nn.relu)(x)
    embeddings = x
    x = Dropout(0.2)(x)
    predictions = Dense(10, activation=tf.nn.softmax)(x)

    '''
    # x = BatchNormalization(trainable=True)(x)
    # x = Flatten()(x)
    x = GlobalAveragePooling2D()(x)
    # and a logistic layer -- let's say we have 200 classes
    # x = Dense(1024, activation='relu')(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dense(256, activation='relu')(x)
    # x = BatchNormalization(trainable=False)(x)
    # x = Dropout(0.2)(x)
    #
    # x = Dropout(0.5)(x)
    x = RBF(10)(x)
    # x = Dense(25)(x)
    # x = Activation('relu')(x)
    embeddings = x
    print(embeddings)
    x = Dense(output_dim=nClasses)(x)
    # x = Dense(output_dim=nClasses)(x)
    prediction = x
    x = Lambda(lambda x: x * 10)(x)
    # x = Dense(nClasses)(x)
    # x = Activation('relu')(x)
    predictions = Activation('softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    return model, base_model.input, embeddings, prediction


def oneHot(a):
    a = a.astype(np.uint8)
    b = np.zeros((a.size, np.unique(a).size))
    b[np.arange(a.size), a] = 1
    return b

def array_to_tensors(data, image_size, batch_size):
    number_of_classes = len(np.unique(data[1]))
    castInt = lambda x: tf.one_hot(tf.cast(x, tf.int32), number_of_classes)
    expand = lambda x: tf.tile(tf.expand_dims(x, -1), [1, 1, 3])
    dataset_images = tf.data.Dataset.from_tensor_slices(data[0]).map(expand)
    dataset_labels = tf.data.Dataset.from_tensor_slices(data[1]).map(castInt)
    dataset = tf.data.Dataset.zip((dataset_images, dataset_labels))
    dataset = dataset.shuffle(buffer_size=4 * batch_size)
    dataset = dataset.repeat(-1)
    dataset = dataset.batch(batch_size).prefetch(1)
    images, labels = dataset.make_one_shot_iterator().get_next()
    images = tf.divide(tf.cast(images, tf.float32), 255.0)
    images.set_shape([batch_size] + image_size + [3])
    labels.set_shape([batch_size] + [number_of_classes])
    images = tf.image.resize(images, image_size)
    # images.set_shape([batch_size] + image_size)
    images.set_shape([batch_size] + image_size + [3])
    return images, labels
    
def array_to_tensors2(data, image_size, batch_size):
    number_of_classes = data[1].shape[1]
    dataset_images = tf.data.Dataset.from_tensor_slices(data[0])
    dataset_labels = tf.data.Dataset.from_tensor_slices(data[1])
    dataset = tf.data.Dataset.zip((dataset_images, dataset_labels))
    dataset = dataset.shuffle(buffer_size=4 * batch_size)
    dataset = dataset.repeat(-1)
    dataset = dataset.batch(batch_size).prefetch(1)
    images, labels = dataset.make_one_shot_iterator().get_next()
    images = tf.divide(tf.cast(images, tf.float32), 255.0)
    images.set_shape([batch_size] + image_size + [3])
    labels.set_shape([batch_size] + [number_of_classes])
    #images = tf.image.resize(images, image_size)
    # images.set_shape([batch_size] + image_size)
    #images.set_shape([batch_size] + image_size + [3])
    return images, labels

def get_embeddings(tf_sess, img_input, embeddings, tensor_data, steps, images=False):
    all_labels = 0
    all_embeddings = 0
    x_train, y_train = tensor_data
    # X.shape[0] // batch_size + 1
    for ii in range(steps):
        _image, _label = tf_sess.run([x_train, y_train])
        _embeddings = tf_sess.run(embeddings, feed_dict={img_input: _image})
        try:
            all_labels = np.concatenate((all_labels, _label), 0)
            all_embeddings = np.concatenate((all_embeddings, _embeddings), 0)
            # print(all_embeddings.shape)
        except:
            all_labels = _label
            all_embeddings = _embeddings
    if images:
        return _image, _label, all_embeddings
    else:
        return all_embeddings, all_labels

def draw_and_save(prototype_index, iteration, all_embeddings, all_labels, cut_constant, title=None):
    distances = (1 - all_embeddings[:, int(prototype_index)])
    distances += np.random.rand(distances.size) / 10
    distances[distances > cut_constant] = cut_constant
    #distances /= cut_constant
    theta = np.random.rand(distances.size) * 2 * np.pi
    X_dist = np.transpose([distances * np.cos(theta), distances * np.sin(theta)])
    # print(X_dist.shape, digits2.target.shape)
    plot_MNIST(X_dist, np.argmax(all_labels, 1), cut_constant, title)
    import os
    if not os.path.exists('clusters/cluster_{:02d}'.format(int(prototype_index))):
        os.makedirs('clusters/cluster_{:02d}'.format(int(prototype_index)))
    plt.savefig('clusters/cluster_{:02d}/iteration_{:02d}.png'.format(int(prototype_index), int(iteration)), bbox_inches='tight')

def draw_and_save_test(prototype_index, all_embeddings, all_labels, test_embedings, test_images, cut_constant, title=None):

    distances = (1 - all_embeddings[:, int(prototype_index)])
    test_embedings = (1 - test_embedings[:, int(prototype_index)])

    distances += np.random.rand(distances.size) / 10
    test_embedings += np.random.rand(test_embedings.size) / 10

    distances[distances > cut_constant] = cut_constant
    test_embedings[test_embedings > cut_constant] = cut_constant
    #distances /= cut_constant
    theta = np.random.rand(distances.size) * 2 * np.pi
    theta2 = np.random.rand(test_embedings.size) * 2 * np.pi

    X_dist = np.transpose([distances * np.cos(theta), distances * np.sin(theta)])
    test_dist = np.transpose([test_embedings * np.cos(theta2), test_embedings * np.sin(theta2)])

    # print(X_dist.shape, digits2.target.shape)
    plot_MNIST2(X_dist, np.argmax(all_labels, 1), [test_dist, test_images], cut_constant, title)
    import os
    if not os.path.exists('test'):
        os.makedirs('test')
    plt.savefig('test/prototype_{:02d}.png'.format(int(prototype_index)), bbox_inches='tight')

def draw_and_save_tsne(iteration, all_embeddings, all_labels, cut_constant, title=None):
    distances = 1 - all_embeddings
    #distances += np.min(distances)
    distances = manifold.TSNE(n_components=2, init='pca').fit_transform(distances)
    distances += np.random.rand(distances.shape[0], distances.shape[1]) / 10
    #cut_constant = np.max(distances)
    distances[distances > cut_constant] = cut_constant
    #for i in centers

    #distances /= cut_constant
    # print(X_dist.shape, digits2.target.shape)
    import os
    if not os.path.exists('tsne'):
        os.makedirs('tsne')
    plot_MNIST(distances, np.argmax(all_labels, 1), cut_constant, title, False)
    plt.savefig('tsne/iteration_{:02d}.png'.format(int(iteration)), bbox_inches='tight')
    print('Figure saved!')

def switch_mode(model, mode='network'):
    if mode == 'rbf':
        for layer in model.layers:
            if len(layer.trainable_weights) > 0:
                if 'rbf' in layer.name:
                    layer.trainable = True
                else:
                    layer.trainable = False
    else:
        for layer in model.layers:
            if len(layer.trainable_weights) > 0:
                if 'rbf' in layer.name:
                    layer.trainable = False
                else:
                    layer.trainable = True
