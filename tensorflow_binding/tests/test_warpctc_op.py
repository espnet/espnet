import tensorflow as tf
import numpy as np
from warpctc_tensorflow import ctc

class WarpCTCTest(tf.test.TestCase):

    def _run_ctc_cpu(self, data, data_lengths,
                     flat_labels, label_lengths,
                     alphabet_size, expected_loss,
                     expected_gradients, expected_error=None):
        self.assertEquals(data.shape, expected_gradients.shape)
        data_t = tf.constant(data)
        data_lengths_t = tf.constant(data_lengths)
        flat_labels_t = tf.constant(flat_labels)
        label_lengths_t = tf.constant(label_lengths)
        loss = ctc(data_t, data_lengths=data_lengths_t,
                   flat_labels=flat_labels_t,
                   label_lengths=label_lengths_t,
                   alphabet_size=alphabet_size)

        grad = tf.gradients(loss, [data_t])[0]
        self.assertShapeEqual(expected_loss, loss)
        self.assertShapeEqual(expected_gradients, grad)
        # Note: using use_gpu=False seems to not work
        # it runs the GPU version instead, which
        # errors out.
        config = tf.ConfigProto(device_count={'GPU': 0})
        with self.test_session(config=config) as sess:
            if expected_error is None:
                (tf_loss, tf_grad) = sess.run([loss, grad])
                self.assertAllClose(tf_loss, expected_loss, atol=1e-6)
                self.assertAllClose(tf_grad, expected_gradients, atol=1e-6)
            else:
                with self.assertRaisesOpError(expected_error):
                    sess.run([loss, grad])

    def _run_ctc_gpu(self, data, data_lengths,
                     flat_labels, label_lengths,
                     alphabet_size, expected_loss,
                     expected_gradients, expected_error=None):
        self.assertEquals(data.shape, expected_gradients.shape)
        data_t = tf.constant(data, dtype=tf.float32)
        data_lengths_t = tf.constant(data_lengths, dtype=tf.int32)
        flat_labels_t = tf.constant(flat_labels, dtype=tf.int32)
        label_lengths_t = tf.constant(label_lengths, dtype=tf.int32)
        loss = ctc(data_t, data_lengths=data_lengths_t,
                   flat_labels=flat_labels_t,
                   label_lengths=label_lengths_t,
                   alphabet_size=alphabet_size)
        grad = tf.gradients(loss, [data_t])[0]
        self.assertShapeEqual(expected_loss, loss)
        self.assertShapeEqual(expected_gradients, grad)
        log_dev_placement = True
        tfconfig = tf.ConfigProto(log_device_placement=log_dev_placement,
                                  allow_soft_placement=False)
        with self.test_session(use_gpu=True, force_gpu=True, config=tfconfig) as sess:
            if expected_error is None:
                (tf_loss, tf_grad) = sess.run([loss, grad])
                self.assertAllClose(tf_loss, expected_loss, atol=1e-6)
                self.assertAllClose(tf_grad, expected_gradients, atol=1e-6)
            else:
                with self.assertRaisesOpError(expected_error):
                    sess.run([loss, grad])

    def test_basic_cpu(self):
        # Softmax activations for the following inputs:
        activations = np.array([
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1]
        ], dtype=np.float32)

        alphabet_size = 5
        # dimensions should be t, n, p: (t timesteps, n minibatches,
        # p prob of each alphabet). This is one instance, so expand
        # dimensions in the middle
        data = np.expand_dims(activations, 1)
        labels = np.asarray([1, 2], dtype=np.int32)
        expected_loss = np.asarray([2.46286], dtype=np.float32)
        gradients = np.asarray([
            [0.177031, -0.708125, 0.177031, 0.177031, 0.177031],
            [0.177031, 0.177031, -0.708125, 0.177031, 0.177031]
        ], dtype=np.float32)
        expected_gradients = np.expand_dims(gradients, 1)
        label_lengths = np.asarray([2], dtype=np.int32)
        data_lengths = np.asarray([2], dtype=np.int32)

        self._run_ctc_cpu(data, data_lengths=data_lengths,
                          flat_labels=labels, label_lengths=label_lengths,
                          alphabet_size=alphabet_size,
                          expected_loss=expected_loss,
                          expected_gradients=expected_gradients)

    def test_basic_gpu(self):
        # Softmax activations for the following inputs:
        activations = np.array([
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1]
        ], dtype=np.float32)

        alphabet_size = 5
        # dimensions should be t, n, p: (t timesteps, n minibatches,
        # p prob of each alphabet). This is one instance, so expand
        # dimensions in the middle
        data = np.expand_dims(activations, 1)
        labels = np.asarray([1, 2], dtype=np.int32)
        expected_loss = np.asarray([2.46286], dtype=np.float32)
        gradients = np.asarray([
            [0.177031, -0.708125, 0.177031, 0.177031, 0.177031],
            [0.177031, 0.177031, -0.708125, 0.177031, 0.177031]
        ], dtype=np.float32)
        expected_gradients = np.expand_dims(gradients, 1)
        label_lengths = np.asarray([2], dtype=np.int32)
        data_lengths = np.asarray([2], dtype=np.int32)
        self._run_ctc_gpu(data, data_lengths=data_lengths,
                          flat_labels=labels, label_lengths=label_lengths,
                          alphabet_size=alphabet_size,
                          expected_loss=expected_loss,
                          expected_gradients=expected_gradients)

    def test_multiple_batches(self):
        activations = np.array([
            [0.1, 0.6, 0.1, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1]
        ], dtype=np.float32)

        alphabet_size = 5
        # dimensions should be t, n, p: (t timesteps, n minibatches,
        # p prob of each alphabet). This is one instance, so expand
        # dimensions in the middle
        _data = np.expand_dims(activations, 1)
        data = np.concatenate([_data, _data[...]], axis=1)
        labels = np.asarray([1, 2, 1, 2], dtype=np.int32)
        expected_loss = np.asarray([2.46286, 2.46286], dtype=np.float32)
        gradients = np.asarray([
            [0.177031, -0.708125, 0.177031, 0.177031, 0.177031],
            [0.177031, 0.177031, -0.708125, 0.177031, 0.177031]
        ], dtype=np.float32)
        _expected_gradients = np.expand_dims(gradients, 1)
        expected_gradients = np.concatenate(
            [_expected_gradients, _expected_gradients[...]], axis=1)

        label_lengths = np.asarray([2, 2], dtype=np.int32)
        data_lengths = np.asarray([2, 2], dtype=np.int32)

        self._run_ctc_cpu(data, data_lengths=data_lengths,
                          flat_labels=labels, label_lengths=label_lengths,
                          alphabet_size=alphabet_size,
                          expected_loss=expected_loss,
                          expected_gradients=expected_gradients)


if __name__ == "__main__":
    tf.test.main()
