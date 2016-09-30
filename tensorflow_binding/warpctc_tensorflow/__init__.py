#import warpctc_tensorflow.kernels

import imp
import tensorflow as tf

lib_file = imp.find_module('kernels', __path__)[1]
tf.load_op_library(lib_file)
