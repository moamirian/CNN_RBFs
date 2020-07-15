import os
import glob
import keras
import numpy as np
from PIL import Image
from scipy.io import loadmat
from sklearn.utils import shuffle

class DatasetProvider():

    def __init__(self, data_dir, dataset, image_size=(224, 224)):
        self.data_dir = data_dir
        self.dataset = dataset
        self.image_size = image_size
        super(DatasetProvider, self).__init__()

    def oneHot(self, a):
        a = a.astype(np.uint8)
        b = np.zeros((a.size, self.number_of_classes), dtype=np.uint8)
        b[np.arange(a.size), a] = 1
        return b

    def read_txt(self, filepath, split_rule):
        list_of_info = []
        with open(filepath, 'r') as file:
            for item in file.readlines():
                list_of_info.append(split_rule(item))
        return list_of_info

    def load_mnist(self):
        dataset_loader = getattr(keras.datasets, self.dataset)
        train_data, test_data = dataset_loader.load_data()
        self.number_of_classes = np.unique(train_data[1]).size        
        x_train, y_train = np.tile(np.expand_dims(train_data[0], -1), [1, 1, 1, 3]), self.oneHot(train_data[1])
        x_test, y_test = np.tile(np.expand_dims(test_data[0], -1), [1, 1, 1, 3]), self.oneHot(test_data[1])       
        return x_train, y_train, x_test, y_test

    def load_cifar(self):
        dataset_loader = getattr(keras.datasets, self.dataset)
        train_data, test_data = dataset_loader.load_data()
        self.number_of_classes = np.unique(train_data[1]).size        
        x_train, y_train = train_data[0], self.oneHot(np.squeeze(train_data[1], 1))
        x_test, y_test = test_data[0], self.oneHot(np.squeeze(test_data[1], 1))
        return x_train, y_train, x_test, y_test

    def load_flowers(self):
        labels = loadmat(os.path.join(*[self.data_dir, self.dataset, 'imagelabels.mat']))['labels'][0]-1
        info = loadmat(os.path.join(*[self.data_dir, self.dataset, 'setid.mat']))
        self.number_of_classes = np.unique(labels).size
        def read_data(ids):
            images, set_labels = np.zeros([len(ids)] + list(self.image_size) + [3], dtype=np.uint8), []
            for i in range(len(ids)):
                im = Image.open(os.path.join(*[self.data_dir, self.dataset, 'images/image_{:05d}.jpg'.format(ids[i])]))
                im = im.resize(self.image_size)
                images[i, :, :, :] = np.array(im)
                set_labels.append(labels[ids[i]-1])
            set_labels = self.oneHot(np.array(set_labels))
            return images, set_labels
        x_train, y_train = read_data(info['trnid'][0])
        x_test, y_test = read_data(info['tstid'][0])
        return x_train, y_train, x_test, y_test

    def load_cub(self):
        split_rule = lambda item: item.split(' ')[1].replace('\n', '')
        labels = self.read_txt(os.path.join(*[self.data_dir, self.dataset, 'image_class_labels.txt']), split_rule)
        labels = [int(item)-1 for item in labels]
        image_dirs = self.read_txt(os.path.join(*[self.data_dir, self.dataset, 'images.txt']), split_rule)
        train_test_split = self.read_txt(os.path.join(*[self.data_dir, self.dataset, 'train_test_split.txt']), split_rule)
        train_idx = [i for i in range(len(train_test_split)) if train_test_split[i]=='1']
        test_idx = [i for i in range(len(train_test_split)) if train_test_split[i]=='0']
        self.number_of_classes = np.unique(np.array(labels)).size
        def read_data(idx_list):
            images, set_labels = np.zeros([len(idx_list)] + list(self.image_size) + [3], dtype=np.uint8), []
            for i in range(len(idx_list)):
                im = Image.open(os.path.join(*[self.data_dir, self.dataset, 'images', image_dirs[idx_list[i]]]))
                im = np.array(im.resize(self.image_size))
                if im.shape[-1] == 3:
                    images[i, :, :, :] = im
                elif im.shape[-1] == 1:
                    images[i, :, :, :] = np.tile(np.expand_dims(im, -1), [1, 1, 1, 3])
                    print('Grayscale image!')
                set_labels.append(labels[idx_list[i]])
            set_labels = self.oneHot(np.array(set_labels))
            return images, set_labels
        x_train, y_train = read_data(train_idx)
        x_test, y_test = read_data(test_idx)
        return x_train, y_train, x_test, y_test

    def load_cars(self):
        # Todo: load the dataset in numpy array
        info = loadmat(os.path.join(*[self.data_dir, self.dataset, 'cars_annos.mat']))['annotations']
        list_of_labels = [info[0][i][-2][0][0]-1 for i in range(info.shape[1])]
        list_of_istrain = [info[0][i][-1][0][0] for i in range(info.shape[1])]
        self.number_of_classes = len(set(list_of_labels))
        def read_data(data_set, set_labels):
            list_of_images = glob.glob(os.path.join(*[self.data_dir, self.dataset, 'cars_{}'.format(data_set), '*.jpg']))
            images = np.zeros([len(list_of_images)] + list(self.image_size) + [3], dtype=np.uint8)
            for i in range(len(list_of_images)):
                im = Image.open(list_of_images[i])
                im = np.array(im.resize(self.image_size))
                if im.shape[-1] == 3:
                    images[i, :, :, :] = im
                elif im.shape[-1] == 1:
                    images[i, :, :, :] = np.tile(np.expand_dims(im, -1), [1, 1, 1, 3])
                    print('Grayscale image!')
            set_labels = self.oneHot(np.array(set_labels))
            return images, set_labels
        set_elements = lambda _list, is_test: [_list[i] for i in range(len(_list)) if list_of_istrain[i]==is_test]
        x_train, y_train = read_data('train', set_elements(list_of_labels, 0))
        x_test, y_test = read_data('test', set_elements(list_of_labels, 1))
        return x_train, y_train, x_test, y_test

    def load_pets(self):
        # Todo: load the dataset in numpy array
        def read_data(data_set):
            list_of_images = self.read_txt(os.path.join(*[self.data_dir, self.dataset, 'annotations',
                                                          '{}.txt'.format(data_set)]), lambda x: x.split(' ')[0])
            set_labels = self.read_txt(os.path.join(*[self.data_dir, self.dataset, 'annotations',
                                                          '{}.txt'.format(data_set)]), lambda x: x.split(' ')[-1].replace('\n', ''))
            set_labels = [int(item)-1 for item in set_labels]
            self.number_of_classes = len(set(set_labels))
            images = np.zeros([len(list_of_images)] + list(self.image_size) + [3], dtype=np.uint8)
            for i in range(len(list_of_images)):
                im = Image.open(os.path.join(*[self.data_dir, self.dataset, 'images', '{}.jpg'.format(list_of_images[i])]))
                im = np.array(im.resize(self.image_size))
                if im.shape[-1] == 3:
                    images[i, :, :, :] = im
                elif im.shape[-1] == 1:
                    images[i, :, :, :] = np.tile(np.expand_dims(im, -1), [1, 1, 1, 3])
                    print('Grayscale image!')
                elif im.shape[-1] == 4:
                    images[i, :, :, :] = im[:, :, :3]
                    print('4D!')
            set_labels = self.oneHot(np.array(set_labels))
            return images, set_labels
        x_train, y_train = read_data('trainval')
        x_test, y_test = read_data('test')
        return x_train, y_train, x_test, y_test

    def load_aircrafts(self):
        classes = self.read_txt(os.path.join(*[self.data_dir, self.dataset, 'data', 'families.txt']), lambda item: item.replace('\n', ''))
        self.number_of_classes = len(classes)
        def read_data(data_set):
            set_labels = self.read_txt(os.path.join(*[self.data_dir, self.dataset, 'data', 'images_family_{}.txt'.format(data_set)]), lambda item: item.replace('\n', ''))
            set_images = ['{:07d}.jpg'.format(int(item.split(' ')[0])) for item in set_labels]
            set_labels = [classes.index(item.split(' ', 1)[1]) for item in set_labels]
            images = np.zeros([len(set_labels)] + list(self.image_size) + [3], dtype=np.uint8)
            for i in range(len(set_images)):
                im = Image.open(os.path.join(*[self.data_dir, self.dataset, 'data', 'images', set_images[i]]))
                im = np.array(im.resize(self.image_size))
                images[i, :, :, :] = im
            set_labels = self.oneHot(np.array(set_labels))
            return images, set_labels
        x_train, y_train = read_data('train')
        x_test, y_test = read_data('test')
        return x_train, y_train, x_test, y_test

    def load_data(self):
        data_path = os.path.join(*[self.data_dir, self.dataset])
        if not os.path.isdir(data_path):
            os.makedirs(data_path)
        try:
            x_train = np.load(os.path.join(*[data_path, 'x_train.npy']))
            y_train = np.load(os.path.join(*[data_path, 'y_train.npy']))
            x_test = np.load(os.path.join(*[data_path, 'x_test.npy']))
            y_test = np.load(os.path.join(*[data_path, 'y_test.npy']))
            self.number_of_classes = y_train.shape[-1]
            self.image_size = x_train.shape[1:3]
            print('Data loaded successfully from saved np arrays!')
        except:
            if 'mnist' in self.dataset.lower():
                x_train, y_train, x_test, y_test = self.load_mnist()
            elif 'cifar' in self.dataset.lower():
                x_train, y_train, x_test, y_test = self.load_cifar()
            elif 'flower' in self.dataset.lower():
                x_train, y_train, x_test, y_test = self.load_flowers()
            elif 'cub' in self.dataset.lower():
                x_train, y_train, x_test, y_test = self.load_cub()
            elif 'pet' in self.dataset.lower():
                x_train, y_train, x_test, y_test = self.load_pets()
            elif 'aircraft' in self.dataset.lower():
                x_train, y_train, x_test, y_test = self.load_aircrafts()
            elif 'car' in self.dataset.lower():
                x_train, y_train, x_test, y_test = self.load_cars()
            np.save(os.path.join(*[data_path, 'x_train']), x_train)
            np.save(os.path.join(*[data_path, 'y_train']), y_train)
            np.save(os.path.join(*[data_path, 'x_test']), x_test)
            np.save(os.path.join(*[data_path, 'y_test']), y_test)
            print('Data loaded successfully from images and saved into np arrays!')
        x_train, y_train = shuffle(x_train, y_train, random_state=10)
        x_test, y_test = shuffle(x_test, y_test, random_state=10)
        return x_train, y_train, x_test, y_test
