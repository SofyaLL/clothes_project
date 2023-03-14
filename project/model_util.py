import os
from collections import defaultdict
from zipfile import ZipFile
# from tensorflow_core.python.keras.utils.data_utils import Sequence

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np

from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.preprocessing import image as process_image
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras import Model
import tensorflow as tf


class DeepModel():
    '''MobileNet deep model.'''
    def __init__(self):
        self._model = self._define_model()

        print('Loading MobileNet.')
        print()

    @staticmethod
    def _define_model(output_layer=-1):
        '''Define a pre-trained MobileNet model.

        Args:
            output_layer: the number of layer that output.

        Returns:
            Class of keras model with weights.
        '''
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.40  # dynamically grow the memory used on the GPU
        # config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        output = base_model.output
        output = GlobalAveragePooling2D()(output)
        model = Model(inputs=base_model.input, outputs=output)
        return model

    @staticmethod
    def preprocess_image(path):
        '''Process an image to numpy array.

        Args:
            path: the path of the image.

        Returns:
            Numpy array of the image.
        '''
        img = process_image.load_img(path, target_size=(224, 224))
        x = process_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x

    @staticmethod
    def cosine_distance(input1, input2):
        '''Calculating the distance of two inputs.

        The return values lies in [-1, 1]. `-1` denotes two features are the most unlike,
        `1` denotes they are the most similar.

        Args:
            input1, input2: two input numpy arrays.

        Returns:
            Element-wise cosine distances of two inputs.
        '''
        # return np.dot(input1, input2) / (np.linalg.norm(input1) * np.linalg.norm(input2))
        return np.dot(input1, input2.T) / \
                np.dot(np.linalg.norm(input1, axis=1, keepdims=True), \
                        np.linalg.norm(input2.T, axis=0, keepdims=True))

    def extract_feature(self, img):
        '''Extract deep feature using MobileNet model.

        Args:
            generator: a predict generator inherit from `keras.utils.Sequence`.

        Returns:
            The output features of all inputs.
        '''
        features = self._model.predict(img, batch_size=1)
        return features


class ImageClassifier():
    def __init__(self):
        self.all_skus = defaultdict(list) # словарь с айди картинками и их features
        self.all_images = {}
        self.model = DeepModel()
        self.top_k = 3

    def extract_features_from_img(self, img_path):
        img = self.model.preprocess_image(img_path)
        feature = self.model.extract_feature(img)
        return feature

    def predict(self, img_path):
        target_features = self.extract_features_from_img(img_path)
        result_dict = {}

        for class_name, features_all in self.all_skus.items():  # (id_img, признаки)
            max_distance = 0
            for features in features_all:
                cur_distance = self.model.cosine_distance(target_features, features)
                cur_distance = cur_distance[0][0]
                if cur_distance > max_distance:
                    max_distance = cur_distance
            result_dict[class_name] = max_distance

        results = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        classes = [r[0] for r in results]
        dists = [float(r[1]) for r in results]
        return classes, dists

    def prepare_archive(self, path_zip, BASE_DIR):
        FILES_DIR = os.path.join(BASE_DIR, "archive")
        broken_images = []
        if not os.path.exists(FILES_DIR):
            os.makedirs(FILES_DIR)

        with ZipFile(path_zip) as f:
            f.extractall(FILES_DIR)
        for class_id in os.listdir(FILES_DIR):
            cur_class_path = os.path.join(FILES_DIR, class_id)
            for img_file in os.listdir(cur_class_path):
                cur_img_path = os.path.join(cur_class_path, img_file)
                try:
                    feature = self.extract_features_from_img(cur_img_path)
                    self.all_skus[class_id].append(feature)
                except:
                    broken_images.append(cur_img_path)

    def prepare_all_images(self, path_zip, BASE_DIR): # [img_path:features]
        broken_images = []
        FILES_DIR = os.path.join(BASE_DIR, "archive")
        if not os.path.exists(FILES_DIR):
            os.makedirs(FILES_DIR)

        with ZipFile(path_zip) as f:
            f.extractall(FILES_DIR)

        for class_id in os.listdir(FILES_DIR):
            cur_class_path = os.path.join(FILES_DIR, class_id)
            for img_file in os.listdir(cur_class_path):
                cur_img_path = os.path.join(cur_class_path, img_file)
                try:
                    feature = self.extract_features_from_img(cur_img_path)
                    self.all_images[cur_img_path] = feature
                except:
                    broken_images.append(cur_img_path)

    def remove_by_id(self, class_id):
        if class_id in self.all_skus:
            self.all_skus.pop(class_id)

    def remove_all(self):
        self.all_skus.clear()






