import copy
from typing import Optional
from typing import Tuple
from typing import Union

import humanfriendly
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.utils.get_default_kwargs import get_default_kwargs

import leaf_audio_pytorch.frontend as frontend

class LeafFrontend(AbsFrontend):
	"""Speech Pretrained Representation frontend structure for ASR."""

	

	def __init__(
		learn_pooling: bool = True,
	    learn_filters: bool = True,
	    n_filters: int = 40,
	    sample_rate: int = 16000,
	    window_len: float = 25.,
	    window_stride: float = 10.,
	    # compression_fn=None,
	    compression_fn=postprocessing.PCENLayer(
	        alpha=0.96,
	        smooth_coef=0.04,
	        delta=2.0,
	        floor=1e-12,
	        trainable=True,
	        learn_smooth_coef=True,
	        per_channel_smooth_coef=True),
	    preemp: bool = False,
	    preemp_init=initializers.PreempInit,
	    complex_conv_init=initializers.GaborInit,
	    pooling_init=initializers.ConstantInit,
	    # regularizer_fn: Optional[tf.keras.regularizers.Regularizer] = None,
	    mean_var_norm: bool = False,
	    spec_augment: bool = False,
	    name='leaf'):

		self.learn_pooling = learn_pooling
	    self.learn_filters = learn_filters 
	    self.n_filters = n_filters
	    self.sample_rate = sample_rate
	    self.window_len = window_len
	    self.window_stride = window_stride
	    # compression_fn=None,
	    self.compression_fn=compression_fn
	    self.preemp = preemp 
	    self.preemp_init = preemp_init
	    self.complex_conv_init = complex_conv_init
	    self.pooling_init= pooling_init
	    # regularizer_fn: Optional[tf.keras.regularizers.Regularizer] = None,
	    self.mean_var_norm = mean_var_norm
	    self.spec_augment = spec_augment
	    self.name=name


		assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

    def output_size(self) -> int:
        return self.n_filters

    def forward(self, input: torch.Tensor):
    	self.leaf = frontend.Leaf(learn_pooling = self.learn_pooling,
	    learn_filters = self.learn_filters,
	    n_filters = self.n_filters,
	    sample_rate = self.sample_rate,
	    window_len = self.window_len,
	    window_stride = self.window_stride,
	    # compression_fn=None,
	    compression_fn=self.compression_fn,
	    preemp = self.preemp,
	    preemp_init=self.preemp_init,
	    complex_conv_init=self.complex_conv_init,
	    pooling_init=self.pooling_init,
	    # regularizer_fn: Optional[tf.keras.regularizers.Regularizer] = self.None,
	    mean_var_norm  = self.mean_var_norm,
	    spec_augment = self.spec_augment,
	    name='leaf')

