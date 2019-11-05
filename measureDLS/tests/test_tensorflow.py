from __future__ import print_function

import os
import time
import unittest

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg, resnet_v2

from measureDLS.measurement.accuracy import Accuracy
from measureDLS.measurement.neuron_coverage import NeuronCoverage
from measureDLS.measurement.robustness import Robustness
from measureDLS.models.tensorflow import TensorFlowModel
from measureDLS.utils import utils


class TestTensorFlow(unittest.TestCase):

    def imagenet_dataset(self, width, height, preprocessing, label_offset):
        dir = utils.python_file_dir(__file__) + '/data/imagenet_val'
        files = sorted(list(filter(lambda file: file.lower().endswith('.jpeg'), os.listdir(dir))))
        num = len(files)
        x = np.empty((num, height, width, 3), dtype=np.float32)
        y = np.empty((num,), dtype=int)
        for i in range(num):
            path = dir + '/' + files[i]
            image_raw = tf.io.read_file(path)
            image = tf.image.decode_image(image_raw)
            if preprocessing == 'vgg16' or preprocessing == 'vgg19':
                image = self._aspect_preserving_resize(image, 256)
                image = self._central_crop([image], height, width)[0]
                image.set_shape([height, width, 3])
                image = tf.to_float(image)
                session = tf.Session()
                image = session.run(image)
                session.close()
            elif preprocessing == 'resnet50_v2' or preprocessing == 'mobilenet_v2':
                image = tf.cast(image, tf.float32)
                session = tf.Session()
                image = session.run(image)
                image = tf.image.central_crop(image, central_fraction=0.875)
                image = tf.expand_dims(image, 0)
                image = tf.image.resize_bilinear(image, [width, height], align_corners=False)
                image = tf.squeeze(image, [0])
                image = session.run(image)
                session.close()
            else:
                raise Exception('Invalid preprocessing', preprocessing)
            x[i] = image
            y[i] = int(files[i].split('_')[-1].split('.')[0]) + label_offset
        return x, y

    def _smallest_size_at_least(self, height, width, smallest_side):
        smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
        height = tf.to_float(height)
        width = tf.to_float(width)
        smallest_side = tf.to_float(smallest_side)
        scale = tf.cond(tf.greater(height, width), lambda: smallest_side / width, lambda: smallest_side / height)
        new_height = tf.to_int32(tf.rint(height * scale))
        new_width = tf.to_int32(tf.rint(width * scale))
        return new_height, new_width

    def _aspect_preserving_resize(self, image, smallest_side):
        smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
        shape = tf.shape(image)
        height = shape[0]
        width = shape[1]
        new_height, new_width = self._smallest_size_at_least(height, width, smallest_side)
        image = tf.expand_dims(image, 0)
        resized_image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
        resized_image = tf.squeeze(resized_image)
        resized_image.set_shape([None, None, 3])
        return resized_image

    def _central_crop(self, image_list, crop_height, crop_width):
        outputs = []
        for image in image_list:
            image_height = tf.shape(image)[0]
            image_width = tf.shape(image)[1]
            offset_height = (image_height - crop_height) / 2
            offset_width = (image_width - crop_width) / 2
            outputs.append(self._crop(image, offset_height, offset_width, crop_height, crop_width))
        return outputs

    def _crop(self, image, offset_height, offset_width, crop_height, crop_width):
        original_shape = tf.shape(image)
        rank_assertion = tf.Assert(tf.equal(tf.rank(image), 3), ['Rank of image must be equal to 3.'])
        with tf.control_dependencies([rank_assertion]):
            cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])
        size_assertion = tf.Assert(
            tf.logical_and(tf.greater_equal(original_shape[0], crop_height), tf.greater_equal(original_shape[1], crop_width)), ['Crop size greater than the image size.'])
        offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))
        with tf.control_dependencies([size_assertion]):
            image = tf.slice(image, offsets, cropped_shape)
        return tf.reshape(image, cropped_shape)

    def cifar10_dataset(self):
        x_test = np.load(utils.python_file_dir(__file__) + '/data/cifar-10-tensorflow/x_test.npy')
        y_test = np.load(utils.python_file_dir(__file__) + '/data/cifar-10-tensorflow/y_test.npy')
        return x_test, y_test

    def mnist_dataset(self):
        x_test = np.load(utils.python_file_dir(__file__) + '/data/MNIST/tensorflow/x_test.npy')
        y_test = np.load(utils.python_file_dir(__file__) + '/data/MNIST/tensorflow/y_test.npy')
        return x_test, y_test

    def test_imagenet_vgg16(self):
        print('*' * 20)
        print('test_imagenet_vgg16')

        session = tf.InteractiveSession(graph=tf.Graph())
        input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        logits, _ = vgg.vgg_16(input, is_training=False)
        restorer = tf.train.Saver()
        restorer.restore(session, utils.python_file_dir(__file__) + '/models/tensorflow_vgg_16/vgg_16.ckpt')
        dataset_original = self.imagenet_dataset(224, 224, 'vgg16', 0)
        x_original = dataset_original[0]
        y_true = dataset_original[1]
        mean = (123.68, 116.78, 103.94)
        std = (1, 1, 1)
        x_preprocessed = (x_original - mean) / std
        dataset_preprocessed = (x_preprocessed, y_true)
        dataset_original_small = (x_original[:5], y_true[:5])
        bounds = (0, 255)

        measure_model = TensorFlowModel(session, logits, input)

        start = time.time()
        accuracy_top_1 = Accuracy(ks=1)
        accuracy_top_5 = Accuracy(ks=5)
        measure_model.predict(dataset_preprocessed, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.get, 0.75)
        self.assertAlmostEqual(accuracy_top_5.get, 0.9)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.4)
        measure_model.intermediate_layer_outputs(x_preprocessed, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.get, 0.12775790101371498)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original_small, bounds, [robustness.update, utils.draw_adversarial_samples], preprocessing=(mean, std))
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

        session.close()

    def test_imagenet_vgg19(self):
        print('*' * 20)
        print('test_imagenet_vgg19')

        session = tf.InteractiveSession(graph=tf.Graph())
        input = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
        logits, _ = vgg.vgg_19(input, is_training=False)
        restorer = tf.train.Saver()
        restorer.restore(session, utils.python_file_dir(__file__) + '/models/tensorflow_vgg_19/vgg_19.ckpt')
        dataset_original = self.imagenet_dataset(224, 224, 'vgg19', 0)
        x_original = dataset_original[0]
        y_true = dataset_original[1]
        mean = (123.68, 116.78, 103.94)
        std = (1, 1, 1)
        x_preprocessed = (x_original - mean) / std
        dataset_preprocessed = (x_preprocessed, y_true)
        dataset_original_small = (x_original[:5], y_true[:5])
        bounds = (0, 255)

        measure_model = TensorFlowModel(session, logits, input)

        start = time.time()
        accuracy_top_1 = Accuracy(ks=1)
        accuracy_top_5 = Accuracy(ks=5)
        measure_model.predict(dataset_preprocessed, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.get, 0.7)
        self.assertAlmostEqual(accuracy_top_5.get, 0.85)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.4)
        measure_model.intermediate_layer_outputs(x_preprocessed, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.get, 0.15575666848121938)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original_small, bounds, [robustness.update, utils.draw_adversarial_samples], preprocessing=(mean, std))
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

        session.close()

    def test_imagenet_resnet50_v2(self):
        print('*' * 20)
        print('test_imagenet_resnet50_v2')

        session = tf.InteractiveSession(graph=tf.Graph())
        input = tf.placeholder(tf.float32, shape=(None, 299, 299, 3))
        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
            resnet_v2.resnet_v2_50(input, num_classes=1001, is_training=False)
        restorer = tf.train.Saver()
        restorer.restore(session, utils.python_file_dir(__file__) + '/models/tensorflow_resnet_v2_50/resnet_v2_50.ckpt')
        logits = session.graph.get_tensor_by_name('resnet_v2_50/predictions/Reshape:0')
        dataset_original = self.imagenet_dataset(299, 299, 'resnet50_v2', 1)
        x_original = dataset_original[0]
        y_true = dataset_original[1]
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        x_preprocessed = (x_original - mean) / std
        dataset_preprocessed = (x_preprocessed, y_true)
        dataset_original_small = (x_original[:5], y_true[:5])
        bounds = (0, 255)

        measure_model = TensorFlowModel(session, logits, input)

        start = time.time()
        accuracy_top_1 = Accuracy(ks=1)
        accuracy_top_5 = Accuracy(ks=5)
        measure_model.predict(dataset_preprocessed, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.get, 0.8)
        self.assertAlmostEqual(accuracy_top_5.get, 0.9)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.4)
        measure_model.intermediate_layer_outputs(x_preprocessed, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.get, 0.47886641839648636)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original_small, bounds, [robustness.update, utils.draw_adversarial_samples], preprocessing=(mean, std))
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

        session.close()

    def test_imagenet_mobilenet_v2(self):
        print('*' * 20)
        print('test_imagenet_mobilenet_v2')

        graph = tf.Graph()
        with tf.gfile.GFile(utils.python_file_dir(__file__) + '/models/tensorflow_mobilenet_v2/mobilenet_v2_1.4_224_frozen.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            input = tf.placeholder(np.float32, shape=[None, 224, 224, 3])
            tf.import_graph_def(graph_def, {'input': input})
        session = tf.InteractiveSession(graph=graph)
        logits = graph.get_tensor_by_name('import/MobilenetV2/Predictions/Reshape_1:0')
        dataset_original = self.imagenet_dataset(224, 224, 'mobilenet_v2', 1)
        x_original = dataset_original[0]
        y_true = dataset_original[1]
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        x_preprocessed = (x_original - mean) / std
        dataset_preprocessed = (x_preprocessed, y_true)
        dataset_original_small = (x_original[:5], y_true[:5])
        bounds = (0, 255)

        measure_model = TensorFlowModel(session, logits, input)

        start = time.time()
        accuracy_top_1 = Accuracy(ks=1)
        accuracy_top_5 = Accuracy(ks=5)
        measure_model.predict(dataset_preprocessed, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.get, 0.85)
        self.assertAlmostEqual(accuracy_top_5.get, 0.9)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.4)
        measure_model.intermediate_layer_outputs(x_preprocessed, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.get, 0.1660512517539325)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_original_small, bounds, [robustness.update, utils.draw_adversarial_samples], preprocessing=(mean, std))
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

        session.close()

    def test_cifar10_simple(self):
        print('*' * 20)
        print('test_cifar10_simple')

        session = tf.InteractiveSession(graph=tf.Graph())
        restorer = tf.train.import_meta_graph(utils.python_file_dir(__file__) + '/models/tensorflow_cifar10_simple/tensorflow_cifar10_simple.meta')
        restorer.restore(session, tf.train.latest_checkpoint(utils.python_file_dir(__file__) + '/models/tensorflow_cifar10_simple/'))
        input = session.graph.get_tensor_by_name('Placeholder:0')
        logits = session.graph.get_tensor_by_name('fc2/add:0')
        dataset = self.cifar10_dataset()
        x = dataset[0]
        y_true = dataset[1]
        dataset_small = (x[:5], y_true[:5])
        bounds = (0, 1)

        measure_model = TensorFlowModel(session, logits, input)

        start = time.time()
        accuracy_top_1 = Accuracy(ks=1)
        accuracy_top_5 = Accuracy(ks=5)
        measure_model.predict(dataset, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.get, 0.3271)
        self.assertAlmostEqual(accuracy_top_5.get, 0.8202)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.6)
        measure_model.intermediate_layer_outputs(x, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.get, 0.1581196581196581)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_small, bounds, [robustness.update, utils.draw_adversarial_samples])
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

    def test_mnist_simple(self):
        print('*' * 20)
        print('test_mnist_simple')

        session = tf.InteractiveSession(graph=tf.Graph())
        restorer = tf.train.import_meta_graph(utils.python_file_dir(__file__) + '/models/tensorflow_mnist_simple/tensorflow_mnist_simple.meta')
        restorer.restore(session, tf.train.latest_checkpoint(utils.python_file_dir(__file__) + '/models/tensorflow_mnist_simple/'))
        input = session.graph.get_tensor_by_name('Placeholder:0')
        logits = session.graph.get_tensor_by_name('fc2/add:0')
        dataset = self.mnist_dataset()
        x = dataset[0]
        y_true = dataset[1]
        dataset_small = (x[:5], y_true[:5])
        bounds = (0, 1)

        measure_model = TensorFlowModel(session, logits, input)

        start = time.time()
        accuracy_top_1 = Accuracy(ks=1)
        accuracy_top_5 = Accuracy(ks=5)
        measure_model.predict(dataset, [accuracy_top_1.update, accuracy_top_5.update])
        self.assertAlmostEqual(accuracy_top_1.get, 0.9377)
        self.assertAlmostEqual(accuracy_top_5.get, 0.9972)
        end = time.time()
        print('accuracy time', end - start)

        neuron_coverage = NeuronCoverage(threshold=0.4)
        measure_model.intermediate_layer_outputs(x, [neuron_coverage.update])
        self.assertAlmostEqual(neuron_coverage.get, 0.3035398230088496)

        start = time.time()
        robustness = Robustness(bounds)
        measure_model.adversarial_samples(dataset_small, bounds, [robustness.update, utils.draw_adversarial_samples])
        self.assertAlmostEqual(robustness.success_rate, 1)
        end = time.time()
        print('robustness time', end - start)

        session.close()


if __name__ == '__main__':
    unittest.main()
