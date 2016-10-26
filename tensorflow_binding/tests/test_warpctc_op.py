import tensorflow as tf
import numpy as np
from warpctc_tensorflow import ctc
from tensorflow.python.client import device_lib

def is_gpu_available():
    """Returns whether TensorFlow can access a GPU."""
    return any(x.device_type == 'GPU' for x in device_lib.list_local_devices())

class WarpCTCTest(tf.test.TestCase):

    def _run_ctc(self, activations, input_lengths,
                 flat_labels, label_lengths,
                 expected_costs, expected_gradients,
                 use_gpu=False, expected_error=None):
        self.assertEquals(activations.shape, expected_gradients.shape)
        activations_t = tf.constant(activations)
        input_lengths_t = tf.constant(input_lengths)
        flat_labels_t = tf.constant(flat_labels)
        label_lengths_t = tf.constant(label_lengths)
        costs = ctc(activations=activations_t,
                    flat_labels=flat_labels_t,
                    label_lengths=label_lengths_t,
                    input_lengths=input_lengths_t)

        grad = tf.gradients(costs, [activations_t])[0]

        self.assertShapeEqual(expected_costs, costs)

        self.assertShapeEqual(expected_gradients, grad)

        log_dev_placement = False
        if not use_gpu:
            # Note: using use_gpu=False seems to not work
            # it runs the GPU version instead
            config = tf.ConfigProto(log_device_placement=log_dev_placement,
                                    device_count={'GPU': 0})
        else:
            config = tf.ConfigProto(log_device_placement=log_dev_placement,
                                    allow_soft_placement=False)

        with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu, config=config) as sess:
            if expected_error is None:
                (tf_costs, tf_grad) = sess.run([costs, grad])
                self.assertAllClose(tf_costs, expected_costs, atol=1e-6)
                self.assertAllClose(tf_grad, expected_gradients, atol=1e-6)
            else:
                with self.assertRaisesOpError(expected_error):
                    sess.run([costs, grad])

                    sess.run([costs, grad])

    def _test_basic(self, use_gpu):
        # Softmax activations for the following inputs:
        activations = np.array([
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1]
        ], dtype=np.float32)

        alphabet_size = 5
        # dimensions should be t, n, p: (t timesteps, n minibatches,
        # p prob of each alphabet). This is one instance, so expand
        # dimensions in the middle
        activations = np.expand_dims(activations, 1)
        labels = np.asarray([1, 2], dtype=np.int32)
        expected_costs = np.asarray([2.46286], dtype=np.float32)
        gradients = np.asarray([
            [0.177031, -0.708125, 0.177031, 0.177031, 0.177031],
            [0.177031, 0.177031, -0.708125, 0.177031, 0.177031]
        ], dtype=np.float32)
        expected_gradients = np.expand_dims(gradients, 1)
        label_lengths = np.asarray([2], dtype=np.int32)
        input_lengths = np.asarray([2], dtype=np.int32)

        self._run_ctc(activations=activations,
                      input_lengths=input_lengths,
                      flat_labels=labels, label_lengths=label_lengths,
                      expected_costs=expected_costs,
                      expected_gradients=expected_gradients,
                      use_gpu=use_gpu)

    def test_basic_cpu(self):
        self._test_basic(use_gpu=False)

    def test_basic_gpu(self):
        if (is_gpu_available()):
            self._test_basic(use_gpu=True)
        else:
            print("Skipping GPU test, no gpus available")

    def _test_multiple_batches(self, use_gpu):
        activations = np.array([
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1]
        ], dtype=np.float32)

        alphabet_size = 5
        # dimensions should be t, n, p: (t timesteps, n minibatches,
        # p prob of each alphabet). This is one instance, so expand
        # dimensions in the middle
        _activations = np.expand_dims(activations, 1)
        activations = np.concatenate([_activations, _activations[...]], axis=1)
        labels = np.asarray([1, 2, 1, 2], dtype=np.int32)
        expected_costs = np.asarray([2.46286, 2.46286], dtype=np.float32)
        gradients = np.asarray([
            [0.177031, -0.708125, 0.177031, 0.177031, 0.177031],
            [0.177031, 0.177031, -0.708125, 0.177031, 0.177031]
        ], dtype=np.float32)
        _expected_gradients = np.expand_dims(gradients, 1)
        expected_gradients = np.concatenate(
            [_expected_gradients, _expected_gradients[...]], axis=1)

        label_lengths = np.asarray([2, 2], dtype=np.int32)
        input_lengths = np.asarray([2, 2], dtype=np.int32)

        self._run_ctc(activations=activations,
                      input_lengths=input_lengths,
                      flat_labels=labels, label_lengths=label_lengths,
                      expected_costs=expected_costs,
                      expected_gradients=expected_gradients,
                      use_gpu=use_gpu)

    def test_multiple_batches_cpu(self):
        self._test_multiple_batches(use_gpu=False)

    def test_multiple_batches_gpu(self):
        if (is_gpu_available()):
            self._test_multiple_batches(use_gpu=True)
        else:
            print("Skipping GPU test, no gpus available")

if __name__ == "__main__":
    tf.test.main()
