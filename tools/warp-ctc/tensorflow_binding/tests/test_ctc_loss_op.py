# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tests for tensorflow.ctc_ops.ctc_decoder_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import warpctc_tensorflow
from tensorflow.python.client import device_lib


def SimpleSparseTensorFrom(x):
  """Create a very simple SparseTensor with dimensions (batch, time).

  Args:
    x: a list of lists of type int

  Returns:
    x_ix and x_val, the indices and values of the SparseTensor<2>.
  """
  x_ix = []
  x_val = []
  for batch_i, batch in enumerate(x):
    for time, val in enumerate(batch):
      x_ix.append([batch_i, time])
      x_val.append(val)
  x_shape = [len(x), np.asarray(x_ix).max(0)[1]+1]
  x_ix = tf.constant(x_ix, tf.int64)
  x_val = tf.constant(x_val, tf.int32)
  x_shape = tf.constant(x_shape, tf.int64)

  return tf.SparseTensor(x_ix, x_val, x_shape)

def is_gpu_available():
  """Returns whether TensorFlow can access a GPU."""
  return any(x.device_type == 'GPU' for x in device_lib.list_local_devices())

class CTCLossTest(tf.test.TestCase):

  def _testCTCLoss(self, inputs, seq_lens, labels,
                   loss_truth, grad_truth,
                   use_gpu=False, expected_err_re=None):
    self.assertEqual(len(inputs), len(grad_truth))

    inputs_t = tf.constant(inputs)

    log_dev_placement = False
    if not use_gpu:
      # Note: using use_gpu=False seems to not work
      # it runs the GPU version instead
      config = tf.ConfigProto(log_device_placement=log_dev_placement,
                              device_count={'GPU': 0})
    else:
      config = tf.ConfigProto(log_device_placement=log_dev_placement,
                              allow_soft_placement=False)

    with tf.get_default_graph()._kernel_label_map({"CTCLoss": "WarpCTC"}):
      with self.test_session(use_gpu=use_gpu, force_gpu=use_gpu, config=config) as sess:
        loss = tf.nn.ctc_loss(inputs=inputs_t,
                              labels=labels,
                              sequence_length=seq_lens)
        grad = tf.gradients(loss, [inputs_t])[0]

        self.assertShapeEqual(loss_truth, loss)
        self.assertShapeEqual(grad_truth, grad)

        if expected_err_re is None:
          (tf_loss, tf_grad) = sess.run([loss, grad])
          self.assertAllClose(tf_loss, loss_truth, rtol=1e-4, atol=1e-4)
          self.assertAllClose(tf_grad, grad_truth, rtol=1e-4, atol=1e-4)
        else:
          with self.assertRaisesOpError(expected_err_re):
            sess.run([loss, grad])

  def _testBasic(self, use_gpu):
    """Test two batch entries."""
    # Input and ground truth from Alex Graves' implementation.
    #
    #### Batch entry 0 #####
    # targets: 0 1 2 1 0
    # outputs:
    # 0 0.633766 0.221185 0.0917319 0.0129757 0.0142857 0.0260553
    # 1 0.111121 0.588392 0.278779 0.0055756 0.00569609 0.010436
    # 2 0.0357786 0.633813 0.321418 0.00249248 0.00272882 0.0037688
    # 3 0.0663296 0.643849 0.280111 0.00283995 0.0035545 0.00331533
    # 4 0.458235 0.396634 0.123377 0.00648837 0.00903441 0.00623107
    # alpha:
    # 0 -3.64753 -0.456075 -inf -inf -inf -inf -inf -inf -inf -inf -inf
    # 1 -inf -inf -inf -0.986437 -inf -inf -inf -inf -inf -inf -inf
    # 2 -inf -inf -inf -inf -inf -2.12145 -inf -inf -inf -inf -inf
    # 3 -inf -inf -inf -inf -inf -inf -inf -2.56174 -inf -inf -inf
    # 4 -inf -inf -inf -inf -inf -inf -inf -inf -inf -3.34211 -inf
    # beta:
    # 0 -inf -2.88604 -inf -inf -inf -inf -inf -inf -inf -inf -inf
    # 1 -inf -inf -inf -2.35568 -inf -inf -inf -inf -inf -inf -inf
    # 2 -inf -inf -inf -inf -inf -1.22066 -inf -inf -inf -inf -inf
    # 3 -inf -inf -inf -inf -inf -inf -inf -0.780373 -inf -inf -inf
    # 4 -inf -inf -inf -inf -inf -inf -inf -inf -inf 0 0
    # prob: -3.34211
    # outputDerivs:
    # 0 -0.366234 0.221185 0.0917319 0.0129757 0.0142857 0.0260553
    # 1 0.111121 -0.411608 0.278779 0.0055756 0.00569609 0.010436
    # 2 0.0357786 0.633813 -0.678582 0.00249248 0.00272882 0.0037688
    # 3 0.0663296 -0.356151 0.280111 0.00283995 0.0035545 0.00331533
    # 4 -0.541765 0.396634 0.123377 0.00648837 0.00903441 0.00623107
    #
    #### Batch entry 1 #####
    #
    # targets: 0 1 1 0
    # outputs:
    # 0 0.30176 0.28562 0.0831517 0.0862751 0.0816851 0.161508
    # 1 0.24082 0.397533 0.0557226 0.0546814 0.0557528 0.19549
    # 2 0.230246 0.450868 0.0389607 0.038309 0.0391602 0.202456
    # 3 0.280884 0.429522 0.0326593 0.0339046 0.0326856 0.190345
    # 4 0.423286 0.315517 0.0338439 0.0393744 0.0339315 0.154046
    # alpha:
    # 0 -1.8232 -1.19812 -inf -inf -inf -inf -inf -inf -inf
    # 1 -inf -2.19315 -2.83037 -2.1206 -inf -inf -inf -inf -inf
    # 2 -inf -inf -inf -2.03268 -3.71783 -inf -inf -inf -inf
    # 3 -inf -inf -inf -inf -inf -4.56292 -inf -inf -inf
    # 4 -inf -inf -inf -inf -inf -inf -inf -5.42262 -inf
    # beta:
    # 0 -inf -4.2245 -inf -inf -inf -inf -inf -inf -inf
    # 1 -inf -inf -inf -3.30202 -inf -inf -inf -inf -inf
    # 2 -inf -inf -inf -inf -1.70479 -0.856738 -inf -inf -inf
    # 3 -inf -inf -inf -inf -inf -0.859706 -0.859706 -0.549337 -inf
    # 4 -inf -inf -inf -inf -inf -inf -inf 0 0
    # prob: -5.42262
    # outputDerivs:
    # 0 -0.69824 0.28562 0.0831517 0.0862751 0.0816851 0.161508
    # 1 0.24082 -0.602467 0.0557226 0.0546814 0.0557528 0.19549
    # 2 0.230246 0.450868 0.0389607 0.038309 0.0391602 -0.797544
    # 3 0.280884 -0.570478 0.0326593 0.0339046 0.0326856 0.190345
    # 4 -0.576714 0.315517 0.0338439 0.0393744 0.0339315 0.154046

    # max_time_steps == 7
    depth = 6

    # seq_len_0 == 5
    targets_0 = [0, 1, 2, 1, 0]
    loss_log_prob_0 = -3.34211
    # dimensions are time x depth
    input_prob_matrix_0 = np.asarray(
        [[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
         [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
         [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
         [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
         [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
        dtype=np.float32)
    input_log_prob_matrix_0 = np.log(input_prob_matrix_0)
    gradient_log_prob_0 = np.asarray(
        [[-0.366234, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
         [0.111121, -0.411608, 0.278779, 0.0055756, 0.00569609, 0.010436],
         [0.0357786, 0.633813, -0.678582, 0.00249248, 0.00272882, 0.0037688],
         [0.0663296, -0.356151, 0.280111, 0.00283995, 0.0035545, 0.00331533],
         [-0.541765, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
        dtype=np.float32)

    # seq_len_1 == 5
    targets_1 = [0, 1, 1, 0]
    loss_log_prob_1 = -5.42262
    # dimensions are time x depth

    input_prob_matrix_1 = np.asarray(
        [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
         [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
         [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
         [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
         [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]],
        dtype=np.float32)
    input_log_prob_matrix_1 = np.log(input_prob_matrix_1)
    gradient_log_prob_1 = np.asarray(
        [[-0.69824, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
         [0.24082, -0.602467, 0.0557226, 0.0546814, 0.0557528, 0.19549],
         [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, -0.797544],
         [0.280884, -0.570478, 0.0326593, 0.0339046, 0.0326856, 0.190345],
         [-0.576714, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]],
        dtype=np.float32)

    # len max_time_steps array of 2 x depth matrices
    inputs = [np.vstack([input_log_prob_matrix_0[t, :],
                         input_log_prob_matrix_1[t, :]])
              for t in range(5)] + 2 * [np.nan*np.ones((2, depth), np.float32)]

    # convert inputs into [max_time x batch_size x depth tensor] Tensor
    inputs = np.asarray(inputs, dtype=np.float32)

    # len batch_size array of label vectors
    labels = SimpleSparseTensorFrom([targets_0, targets_1])

    # batch_size length vector of sequence_lengths
    seq_lens = np.array([5, 5], dtype=np.int32)

    # output: batch_size length vector of negative log probabilities
    loss_truth = np.array([-loss_log_prob_0, -loss_log_prob_1], np.float32)

    # output: len max_time_steps array of 2 x depth matrices
    grad_truth = [np.vstack([gradient_log_prob_0[t, :],
                             gradient_log_prob_1[t, :]])
                  for t in range(5)] + 2 * [np.zeros((2, depth), np.float32)]

    # convert grad_truth into [max_time x batch_size x depth] Tensor
    grad_truth = np.asarray(grad_truth, dtype=np.float32)

    self._testCTCLoss(inputs, seq_lens, labels, loss_truth, grad_truth, use_gpu=use_gpu)

  def testBasicCPU(self):
    self._testBasic(use_gpu=False)

  def testBasicGPU(self):
    if (is_gpu_available()):
      self._testBasic(use_gpu=True)
    else:
      print("Skipping GPU test, no gpus available")

if __name__ == "__main__":
  tf.test.main()
