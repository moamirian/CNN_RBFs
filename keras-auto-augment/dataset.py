import random
from keras.preprocessing.image import ImageDataGenerator
from autoaugment import *
from PIL import Image
from auto_augment import cutout, apply_policy
from utils import *


class Cifar10ImageDataGenerator:
    def __init__(self, args):
        self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, fill_mode='constant', cval=0, horizontal_flip=True)

        self.means = np.array([0.4914009 , 0.48215896, 0.4465308])
        self.stds = np.array([0.24703279, 0.24348423, 0.26158753])

        self.args = args
        if args.auto_augment:
            self.image_policy = CIFAR10Policy()
            self.policies = self.image_policy.policies

    def standardize(self, x):
        x = x.astype('float32') / 255

        means = self.means.reshape(1, 1, 1, 3)
        stds = self.stds.reshape(1, 1, 1, 3)

        x -= means
        x /= (stds + 1e-6)

        return x

    def flow(self, x, y=None, batch_size=32, shuffle=True, sample_weight=None,
             seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None):
        batches = self.datagen.flow(x, y, batch_size, shuffle, sample_weight,
                               seed, save_to_dir, save_prefix, save_format, subset)

        while True:
            x_batch, y_batch = next(batches)

            if self.args.cutout:
                for i in range(x_batch.shape[0]):
                    x_batch[i] = cutout(x_batch[i])

            if self.args.auto_augment:
                x_batch = x_batch.astype('uint8')
                for i in range(x_batch.shape[0]):             
                    x_batch[i] = np.array(self.image_policy(Image.fromarray(x_batch[i])))

            x_batch = self.standardize(x_batch)

            yield x_batch, y_batch
            
class ImageNetImageDataGenerator(Cifar10ImageDataGenerator):
    def __init__(self, args):
        self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, fill_mode='constant', cval=0, horizontal_flip=True)

        self.means = np.array([0.4914009 , 0.48215896, 0.4465308])
        self.stds = np.array([0.24703279, 0.24348423, 0.26158753])

        self.args = args
        if args.auto_augment:
            self.image_policy = ImageNetPolicy()
            self.policies = self.image_policy.policies
            
class SVHNImageDataGenerator(Cifar10ImageDataGenerator):
    def __init__(self, args):
        self.datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, fill_mode='constant', cval=0, horizontal_flip=True)

        self.means = np.array([0.4914009 , 0.48215896, 0.4465308])
        self.stds = np.array([0.24703279, 0.24348423, 0.26158753])

        self.args = args
        if args.auto_augment:
            self.image_policy = SVHNPolicy()
            self.policies = self.image_policy.policies

def main():
    import argparse
    import matplotlib.pyplot as plt
    from keras.datasets import cifar10

    parser = argparse.ArgumentParser()
    parser.add_argument('--cutout', default=True, type=str2bool)
    parser.add_argument('--auto-augment', default=True, type=str2bool)
    args = parser.parse_args()

    datagen = Cifar10ImageDataGenerator(args)

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    for imgs, _ in datagen.flow(x_train, y_train):
        plt.imshow(imgs[0].astype('uint8'))
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
