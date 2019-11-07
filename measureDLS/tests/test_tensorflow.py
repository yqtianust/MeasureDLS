from __future__ import print_function

import unittest
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg, resnet_v2

from measureDLS.measurement.accuracy import Accuracy
from measureDLS.measurement.neuron_coverage import NeuronCoverage
from measureDLS.measurement.robustness import Robustness
from measureDLS.models.tensorflow import TensorFlowModel
from measureDLS.utils import utils


class TestTensorFlow(unittest.TestCase):
    class ImageNetValData():

        class ImageNetValDataX():

            def __init__(self, dir, filenames, width, height, fashion, transform):
                self._dir = dir
                self._filenames = filenames
                self._width = width
                self._height = height
                self._fashion = fashion
                self._transform = transform

            def __len__(self):
                return len(self._filenames)

            def __getitem__(self, index):
                session = tf.compat.v1.Session()
                x = None
                for filename in self._filenames[index]:
                    path = self._dir + '/' + filename
                    image = tf.image.decode_image(tf.io.read_file(path), channels=3)
                    image = session.run(image)
                    if self._fashion == 'vgg16' or self._fashion == 'vgg19':
                        image = self._aspect_preserving_resize(image, 256)
                        image = self._central_crop([image], self._height, self._width)[0]
                        image.set_shape([self._height, self._width, 3])
                        image = tf.cast(image, dtype=tf.float32)
                        image = session.run(image)
                    elif self._fashion == 'resnet50_v2' or self._fashion == 'mobilenet_v2':
                        image = tf.cast(image, tf.float32)
                        image = session.run(image)
                        image = tf.image.central_crop(image, central_fraction=0.875)
                        image = tf.expand_dims(image, 0)
                        image = tf.compat.v1.image.resize_bilinear(image, [self._width, self._height], align_corners=False)
                        image = tf.squeeze(image, [0])
                        image = session.run(image)
                    else:
                        raise Exception('Invalid fashion', self._fashion)
                    if self._transform is not None:
                        image = self._transform(image)
                    image = np.expand_dims(image, axis=0)
                    if x is None:
                        x = image
                    else:
                        x = np.concatenate((x, image))
                session.close()
                return x

            def _smallest_size_at_least(self, height, width, smallest_side):
                smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
                height = tf.cast(height, dtype=tf.float32)
                width = tf.cast(width, dtype=tf.float32)
                smallest_side = tf.cast(smallest_side, dtype=tf.float32)
                scale = tf.cond(tf.greater(height, width), lambda: smallest_side / width, lambda: smallest_side / height)
                new_height = tf.cast(tf.math.rint(height * scale), dtype=tf.int32)
                new_width = tf.cast(tf.math.rint(width * scale), dtype=tf.int32)
                return new_height, new_width

            def _aspect_preserving_resize(self, image, smallest_side):
                smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
                shape = tf.shape(image)
                height = shape[0]
                width = shape[1]
                new_height, new_width = self._smallest_size_at_least(height, width, smallest_side)
                image = tf.expand_dims(image, 0)
                resized_image = tf.compat.v1.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
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
                offsets = tf.cast(tf.stack([offset_height, offset_width, 0]), dtype=tf.int32)
                with tf.control_dependencies([size_assertion]):
                    image = tf.slice(image, offsets, cropped_shape)
                return tf.reshape(image, cropped_shape)

        def __init__(self, width, height, fashion, transform=None, label_offset=0):
            dir = utils.python_file_dir(__file__) + '/data/imagenet_val'
            filenames = []
            self.y = []
            with open(dir + '/' + 'ILSVRC2012_validation_ground_truth.txt', 'r') as f:
                lines = f.readlines()
            for line in lines:
                splits = line.split('---')
                if len(splits) != 5:
                    continue
                filenames.append(splits[0])
                self.y.append(int(splits[2]))
            self.x = self.ImageNetValDataX(dir, filenames, width, height, fashion, transform)
            self.y = np.array(self.y, dtype=int) + label_offset

    def cifar10_data(self):
        x_test = np.load(utils.python_file_dir(__file__) + '/data/cifar-10-tensorflow/x_test.npy')
        y_test = np.load(utils.python_file_dir(__file__) + '/data/cifar-10-tensorflow/y_test.npy')
        return x_test, y_test

    def mnist_data(self):
        x_test = np.load(utils.python_file_dir(__file__) + '/data/MNIST/tensorflow/x_test.npy')
        y_test = np.load(utils.python_file_dir(__file__) + '/data/MNIST/tensorflow/y_test.npy')
        return x_test, y_test

    def test_imagenet_vgg16(self):
        tf.get_logger().setLevel('ERROR')
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            logits, _ = vgg.vgg_16(input, is_training=False)
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, utils.python_file_dir(__file__) + '/models/tensorflow_vgg_16/vgg_16.ckpt')
        mean = (123.68, 116.78, 103.94)
        std = (1, 1, 1)
        data_preprocess = self.ImageNetValData(224, 224, 'vgg16', transform=lambda x: (x - mean) / std, label_offset=0)
        data_original = self.ImageNetValData(224, 224, 'vgg16', transform=None, label_offset=0)
        bounds = (0, 255)

        measure_model = TensorFlowModel(session, logits, input)

        accuracy = Accuracy()
        measure_model.predict(data_preprocess.x, data_preprocess.y, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(data_preprocess.x, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(data_original.x, data_original.y, 3, bounds, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1, preprocessing=(mean, std))

        session.close()

        self.assertAlmostEqual(accuracy.get(1), 0.600000)
        self.assertAlmostEqual(accuracy.get(5), 0.925000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.630143, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_imagenet_vgg19(self):
        tf.get_logger().setLevel('ERROR')
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 224, 224, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            logits, _ = vgg.vgg_19(input, is_training=False)
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, utils.python_file_dir(__file__) + '/models/tensorflow_vgg_19/vgg_19.ckpt')
        mean = (123.68, 116.78, 103.94)
        std = (1, 1, 1)
        data_preprocess = self.ImageNetValData(224, 224, 'vgg19', transform=lambda x: (x - mean) / std, label_offset=0)
        data_original = self.ImageNetValData(224, 224, 'vgg19', transform=None, label_offset=0)
        bounds = (0, 255)

        measure_model = TensorFlowModel(session, logits, input)

        accuracy = Accuracy()
        measure_model.predict(data_preprocess.x, data_preprocess.y, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(data_preprocess.x, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(data_original.x, data_original.y, 3, bounds, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1, preprocessing=(mean, std))

        session.close()

        self.assertAlmostEqual(accuracy.get(1), 0.625000)
        self.assertAlmostEqual(accuracy.get(5), 0.925000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.576892, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_imagenet_resnet50_v2(self):
        tf.get_logger().setLevel('ERROR')
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        input = tf.compat.v1.placeholder(tf.float32, shape=(None, 299, 299, 3))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=DeprecationWarning)
            with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope()):
                resnet_v2.resnet_v2_50(input, num_classes=1001, is_training=False)
        restorer = tf.compat.v1.train.Saver()
        restorer.restore(session, utils.python_file_dir(__file__) + '/models/tensorflow_resnet_v2_50/resnet_v2_50.ckpt')
        logits = session.graph.get_tensor_by_name('resnet_v2_50/predictions/Reshape:0')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_preprocess = self.ImageNetValData(299, 299, 'resnet50_v2', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = self.ImageNetValData(299, 299, 'resnet50_v2', transform=None, label_offset=1)
        bounds = (0, 255)

        measure_model = TensorFlowModel(session, logits, input)

        accuracy = Accuracy()
        measure_model.predict(data_preprocess.x, data_preprocess.y, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(data_preprocess.x, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(data_original.x, data_original.y, 3, bounds, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1, preprocessing=(mean, std))

        session.close()

        self.assertAlmostEqual(accuracy.get(1), 0.750000)
        self.assertAlmostEqual(accuracy.get(5), 0.875000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.600558, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_imagenet_mobilenet_v2(self):
        tf.get_logger().setLevel('ERROR')
        graph = tf.Graph()
        with tf.io.gfile.GFile(utils.python_file_dir(__file__) + '/models/tensorflow_mobilenet_v2/mobilenet_v2_1.4_224_frozen.pb', 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        with graph.as_default():
            input = tf.compat.v1.placeholder(np.float32, shape=[None, 224, 224, 3])
            tf.import_graph_def(graph_def, {'input': input})
        session = tf.compat.v1.InteractiveSession(graph=graph)
        logits = graph.get_tensor_by_name('import/MobilenetV2/Predictions/Reshape_1:0')
        mean = (127.5, 127.5, 127.5)
        std = (127.5, 127.5, 127.5)
        data_preprocess = self.ImageNetValData(224, 224, 'mobilenet_v2', transform=lambda x: (x - mean) / std, label_offset=1)
        data_original = self.ImageNetValData(224, 224, 'mobilenet_v2', transform=None, label_offset=1)
        bounds = (0, 255)

        measure_model = TensorFlowModel(session, logits, input)

        accuracy = Accuracy()
        measure_model.predict(data_preprocess.x, data_preprocess.y, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(data_preprocess.x, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(data_original.x, data_original.y, 3, bounds, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1, preprocessing=(mean, std))

        session.close()

        self.assertAlmostEqual(accuracy.get(1), 0.725000)
        self.assertAlmostEqual(accuracy.get(5), 0.900000)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.288900, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_cifar10_simple(self):
        tf.get_logger().setLevel('ERROR')
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        restorer = tf.compat.v1.train.import_meta_graph(utils.python_file_dir(__file__) + '/models/tensorflow_cifar10_simple/tensorflow_cifar10_simple.meta')
        restorer.restore(session, tf.train.latest_checkpoint(utils.python_file_dir(__file__) + '/models/tensorflow_cifar10_simple/'))
        input = session.graph.get_tensor_by_name('Placeholder:0')
        logits = session.graph.get_tensor_by_name('fc2/add:0')
        x, y = self.cifar10_data()
        bounds = (0, 1)

        measure_model = TensorFlowModel(session, logits, input)

        accuracy = Accuracy()
        measure_model.predict(x, y, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(x, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(x, y, 3, bounds, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1)

        session.close()

        self.assertAlmostEqual(accuracy.get(1), 0.327100)
        self.assertAlmostEqual(accuracy.get(5), 0.820200)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.551282, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)

    def test_mnist_simple(self):
        tf.get_logger().setLevel('ERROR')
        session = tf.compat.v1.InteractiveSession(graph=tf.Graph())
        restorer = tf.compat.v1.train.import_meta_graph(utils.python_file_dir(__file__) + '/models/tensorflow_mnist_simple/tensorflow_mnist_simple.meta')
        restorer.restore(session, tf.train.latest_checkpoint(utils.python_file_dir(__file__) + '/models/tensorflow_mnist_simple/'))
        input = session.graph.get_tensor_by_name('Placeholder:0')
        logits = session.graph.get_tensor_by_name('fc2/add:0')
        x, y = self.mnist_data()
        bounds = (0, 1)

        measure_model = TensorFlowModel(session, logits, input)

        accuracy = Accuracy()
        measure_model.predict(x, y, [accuracy.update, accuracy.report])

        neuron_coverage = NeuronCoverage()
        measure_model.intermediate_layer_outputs(x, [neuron_coverage.update, neuron_coverage.report])

        robustness = Robustness(bounds)
        measure_model.adversarial_samples(x, y, 3, bounds, [robustness.update, robustness.report, utils.draw_adversarial_samples], batch_size=1)

        session.close()

        self.assertAlmostEqual(accuracy.get(1), 0.937700)
        self.assertAlmostEqual(accuracy.get(5), 0.997200)
        self.assertAlmostEqual(neuron_coverage.get(0.3), 0.591150, places=2)
        self.assertAlmostEqual(robustness.success_rate, 1.000000)


if __name__ == '__main__':
    unittest.main()
