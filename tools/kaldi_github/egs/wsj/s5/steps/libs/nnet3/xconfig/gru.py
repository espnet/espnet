# Copyright 2016    Johns Hopkins University (Dan Povey)
#           2017    Gaofeng Cheng (UCAS)
#           2017    Lu Huang (THU)
# Apache 2.0.


""" This module has the implementations of different GRU layers.
"""
from __future__ import print_function
import math
import re
import sys
from libs.nnet3.xconfig.basic_layers import XconfigLayerBase

# This class is for lines like
#   'gru-layer name=gru1 input=[-1] delay=-3'
# It generates an GRU sub-graph without output projections.
# The output dimension of the layer may be specified via 'cell-dim=xxx', but if not specified,
# the dimension defaults to the same as the input.
# See other configuration values below.
# decay-time is deprecated under GRU or PGRU, as I found the PGRUs do not need the decay-time option to get generalized to unseen sequence length
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   cell-dim=-1              [Dimension of the cell]
#   delay=-1                 [Delay in the recurrent connections of the GRU/LSTM ]
#   clipping-threshold=30    [similar to LSTMs ,nnet3 GRUs use a gradient clipping component at the recurrent connections.
#                             This is the threshold used to decide if clipping has to be activated ]
#   zeroing-interval=20      [interval at which we (possibly) zero out the recurrent derivatives.]
#   zeroing-threshold=15     [We only zero out the derivs every zeroing-interval, if derivs exceed this value.]
#   self-repair-scale-nonlinearity=1e-5      [It is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent]
#                                       i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent ]
#   ng-per-element-scale-options=''     [Additional options used for the diagonal matrices in the GRU/LSTM ]
#   ng-affine-options=''                [Additional options used for the full matrices in the GRU/LSTM, can be used to do things like set biases to initialize to 1]
class XconfigGruLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "gru-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input':'[-1]',
                        'cell-dim' : -1, # this is a compulsory argument
                        'clipping-threshold' : 30.0,
                        'delay' : -1,
                        'ng-per-element-scale-options' : ' max-change=0.75',
                        'ng-affine-options' : ' max-change=0.75 ',
                        'self-repair-scale-nonlinearity' : 0.00001,
                        'zeroing-interval' : 20,
                        'zeroing-threshold' : 15.0
                        }

    def set_derived_configs(self):
        if self.config['cell-dim'] <= 0:
            self.config['cell-dim'] = self.descriptors['input']['dim']

    def check_configs(self):
        key = 'cell-dim'
        if self.config['cell-dim'] <= 0:
            raise RuntimeError("cell-dim has invalid value {0}.".format(self.config[key]))

        if self.config['delay'] == 0:
            raise RuntimeError("delay cannot be zero")

        for key in ['self-repair-scale-nonlinearity']:
            if self.config[key] < 0.0 or self.config[key] > 1.0:
                raise RuntimeError("{0} has invalid value {1}.".format(key, self.config[key]))

    def output_name(self, auxiliary_output = None):
        node_name = 's_t'
        return '{0}.{1}'.format(self.name, node_name)

    def output_dim(self, auxiliary_output = None):
        return self.config['cell-dim']

    def get_full_config(self):
        ans = []
        config_lines = self.generate_gru_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in LSTM initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the LSTM config
    def generate_gru_config(self):

        # assign some variables to reduce verbosity
        name = self.name
        # in the below code we will just call descriptor_strings as descriptors for conciseness
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        cell_dim = self.config['cell-dim']
        delay = self.config['delay']
        bptrunc_str = ("clipping-threshold={0}"
                      " zeroing-threshold={1}"
                      " zeroing-interval={2}"
                      " recurrence-interval={3}"
                      "".format(self.config['clipping-threshold'],
                                self.config['zeroing-threshold'],
                                self.config['zeroing-interval'], abs(delay)))
        repair_nonlin = self.config['self-repair-scale-nonlinearity']
        repair_nonlin_str = "self-repair-scale={0:.10f}".format(repair_nonlin) if repair_nonlin is not None else ''
        affine_str = self.config['ng-affine-options']
        # Natural gradient per element scale parameters
        # TODO: decide if we want to keep exposing these options
        ng_per_element_scale_options = self.config['ng-per-element-scale-options']
        if re.search('param-mean', ng_per_element_scale_options) is None and \
           re.search('param-stddev', ng_per_element_scale_options) is None:
           ng_per_element_scale_options += " param-mean=0.0 param-stddev=1.0 "
        pes_str = ng_per_element_scale_options

        # formulation like:
        # z_t = \sigmoid ( x_t * U^z + h_{t-1} * W^z ) // update gate
        # r_t = \sigmoid ( x_t * U^r + h_{t-1} * W^r ) // reset gate
        # \tilde{h}_t = \tanh ( x_t * U^h + ( h_{t-1} \dot r_t ) * W^h )
        # h_t = ( 1 - z_t ) \dot \tilde{h}_t + z_t \dot h_{t-1}
        # y_t = h_t // y_t is the output

        configs = []
        configs.append("# Update gate control : W_z* matrics")
        configs.append("component name={0}.W_z.xs_z type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + cell_dim, cell_dim, affine_str))
        
        configs.append("# Reset gate control : W_r* matrics")
        configs.append("component name={0}.W_z.xs_r type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + cell_dim, cell_dim, affine_str))

        configs.append("# h related matrix : W_h* matrics")
        configs.append("component name={0}.W_h.UW type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + cell_dim, cell_dim , affine_str))
        
        configs.append("# Defining the non-linearities")
        configs.append("component name={0}.z type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.r type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.h type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))

        configs.append("# Defining the components for other cell computations")
        configs.append("component name={0}.h1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.y1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.y2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.y type=NoOpComponent dim={1}".format(name, cell_dim))

        recurrent_connection = '{0}.s_t'.format(name)

        configs.append("# z_t")
        configs.append("component-node name={0}.z_t_pre component={0}.W_z.xs_z input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.z_t component={0}.z input={0}.z_t_pre".format(name))

        configs.append("# r_t")
        configs.append("component-node name={0}.r_t_pre component={0}.W_z.xs_r input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.r_t component={0}.r input={0}.r_t_pre".format(name))
        
        configs.append("# h_t")
        configs.append("component-node name={0}.h1_t component={0}.h1 input=Append({0}.r_t, IfDefined(Offset({1}, {2})))".format(name, recurrent_connection, delay))
        configs.append("component-node name={0}.h_t_pre component={0}.W_h.UW input=Append({1}, {0}.h1_t)".format(name, input_descriptor))
        configs.append("component-node name={0}.h_t component={0}.h input={0}.h_t_pre".format(name))
        
        configs.append("# y_t")
        configs.append("# The following two lines are to implement (1 - z_t)")
        configs.append("component-node name={0}.y1_t component={0}.y1 input=Append({0}.h_t, Sum(Scale(-1.0,{0}.z_t), Const(1.0, {1})))".format(name, cell_dim))
        configs.append("component-node name={0}.y2_t component={0}.y2 input=Append(IfDefined(Offset({1}, {2})), {0}.z_t)".format(name, recurrent_connection, delay))
        configs.append("component-node name={0}.y_t component={0}.y input=Sum({0}.y1_t, {0}.y2_t)".format(name))

        configs.append("# s_t : recurrence")
        configs.append("component name={0}.s_r type=BackpropTruncationComponent dim={1} {2}".format(name, cell_dim, bptrunc_str))

        configs.append("# s_t will be output and recurrence")
        configs.append("component-node name={0}.s_t component={0}.s_r input={0}.y_t".format(name))
        return configs


# This class is for lines like
#   'pgru-layer name=pgru1 input=[-1] delay=-3'
# It generates an PGRU sub-graph with output projections. It can also generate
# outputs without projection, but you could use the XconfigGruLayer for this
# simple RNN.
# The output dimension of the layer may be specified via 'cell-dim=xxx', but if not specified,
# the dimension defaults to the same as the input.
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   cell-dim=-1            [Dimension of the cell]
#   recurrent-projection-dim [Dimension of the projection used in recurrent connections, e.g. cell-dim/4]
#   non-recurrent-projection-dim   [Dimension of the projection in non-recurrent connections,
#                                   in addition to recurrent-projection-dim, e.g. cell-dim/4]
#   delay=-1                 [Delay in the recurrent connections of the GRU ]
#   clipping-threshold=30    [nnet3 GRU use a gradient clipping component at the recurrent connections.
#                             This is the threshold used to decide if clipping has to be activated ]
#   zeroing-interval=20      [interval at which we (possibly) zero out the recurrent derivatives.]
#   zeroing-threshold=15     [We only zero out the derivs every zeroing-interval, if derivs exceed this value.]
#   self-repair-scale-nonlinearity=1e-5      [It is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent]
#                                       i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent ]
#   ng-per-element-scale-options=''   [Additional options used for the diagonal matrices in the GRU ]
#   ng-affine-options=''              [Additional options used for the full matrices in the GRU, can be used to do things like set biases to initialize to 1]

class XconfigPgruLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "pgru-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input' : '[-1]',
                        'cell-dim' : -1, # this is a compulsory argument
                        'recurrent-projection-dim' : -1,  # defaults to cell-dim / 4
                        'non-recurrent-projection-dim' : -1, # defaults to
                                                             # recurrent-projection-dim
                        'clipping-threshold' : 30.0,
                        'delay' : -1,
                        'ng-per-element-scale-options' : ' max-change=0.75 ',
                        'ng-affine-options' : ' max-change=0.75 ',
                        'self-repair-scale-nonlinearity' : 0.00001,
                        'zeroing-interval' : 20,
                        'zeroing-threshold' : 15.0
                       }

    def set_derived_configs(self):
        if self.config['recurrent-projection-dim'] <= 0:
            self.config['recurrent-projection-dim'] = self.config['cell-dim'] / 4

        if self.config['non-recurrent-projection-dim'] <= 0:
            self.config['non-recurrent-projection-dim'] = \
               self.config['recurrent-projection-dim']

    def check_configs(self):
        for key in ['cell-dim', 'recurrent-projection-dim',
                    'non-recurrent-projection-dim']:
            if self.config[key] <= 0:
                raise RuntimeError("{0} has invalid value {1}.".format(
                    key, self.config[key]))

        if self.config['delay'] == 0:
            raise RuntimeError("delay cannot be zero")

        if (self.config['recurrent-projection-dim'] +
            self.config['non-recurrent-projection-dim'] >
            self.config['cell-dim']):
            raise RuntimeError("recurrent+non-recurrent projection dim exceeds "
                                "cell dim.")
        for key in ['self-repair-scale-nonlinearity']:
            if self.config[key] < 0.0 or self.config[key] > 1.0:
                raise RuntimeError("{0} has invalid value {2}."
                                   .format(self.layer_type, key,
                                           self.config[key]))

    def auxiliary_outputs(self):
        return ['h_t']

    def output_name(self, auxiliary_output = None):
        node_name = 'sn_t'
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                node_name = auxiliary_output
            else:
                raise Exception("In {0} of type {1}, unknown auxiliary output name {1}".format(self.layer_type, auxiliary_output))

        return '{0}.{1}'.format(self.name, node_name)

    def output_dim(self, auxiliary_output = None):
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                if node_name == 'c_t':
                    return self.config['cell-dim']
                # add code for other auxiliary_outputs here when we decide to expose them
            else:
                raise Exception("In {0} of type {1}, unknown auxiliary output name {1}".format(self.layer_type, auxiliary_output))

        return self.config['recurrent-projection-dim'] + self.config['non-recurrent-projection-dim']

    def get_full_config(self):
        ans = []
        config_lines = self.generate_pgru_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in LSTM initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the PGRU config
    def generate_pgru_config(self):

        # assign some variables to reduce verbosity
        name = self.name
        # in the below code we will just call descriptor_strings as descriptors for conciseness
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        cell_dim = self.config['cell-dim']
        rec_proj_dim = self.config['recurrent-projection-dim']
        nonrec_proj_dim = self.config['non-recurrent-projection-dim']
        delay = self.config['delay']
        repair_nonlin = self.config['self-repair-scale-nonlinearity']
        repair_nonlin_str = "self-repair-scale={0:.10f}".format(repair_nonlin) if repair_nonlin is not None else ''
        bptrunc_str = ("clipping-threshold={0}"
                      " zeroing-threshold={1}"
                      " zeroing-interval={2}"
                      " recurrence-interval={3}"
                      "".format(self.config['clipping-threshold'],
                                self.config['zeroing-threshold'],
                                self.config['zeroing-interval'],
                                abs(delay)))
        affine_str = self.config['ng-affine-options']
        pes_str = self.config['ng-per-element-scale-options']

        # Natural gradient per element scale parameters
        # TODO: decide if we want to keep exposing these options
        if re.search('param-mean', pes_str) is None and \
           re.search('param-stddev', pes_str) is None:
           pes_str += " param-mean=0.0 param-stddev=1.0 "

        # formulation like:
        # z_t = \sigmoid ( x_t * U^z + s_{t-1} * W^z ) // update gate
        # r_t = \sigmoid ( x_t * U^r + s_{t-1} * W^r ) // reset gate
        # \tilde{h}_t = \tanh ( x_t * U^h + ( s_{t-1} \dot r_t ) * W^h )
        # h_t = ( 1 - z_t ) \dot \tilde{h}_t + z_t \dot h_{t-1}
        # y_t = h_t * W^y
        # s_t = y_t (0:rec_proj_dim-1)
        
        configs = []
        configs.append("# Update gate control : W_z* matrics")
        configs.append("component name={0}.W_z.xs_z type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, cell_dim, affine_str))
        
        configs.append("# Reset gate control : W_r* matrics")
        configs.append("component name={0}.W_z.xs_r type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, rec_proj_dim, affine_str))

        configs.append("# h related matrix : W_h* matrics")
        configs.append("component name={0}.W_h.UW type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, cell_dim , affine_str))
        
        configs.append("# Defining the non-linearities")
        configs.append("component name={0}.z type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.r type=SigmoidComponent dim={1} {2}".format(name, rec_proj_dim, repair_nonlin_str))
        configs.append("component name={0}.h type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))

        configs.append("# Defining the components for other cell computations")
        configs.append("component name={0}.h1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * rec_proj_dim, rec_proj_dim))
        configs.append("component name={0}.y1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.y2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.y type=NoOpComponent dim={1}".format(name, cell_dim))

        recurrent_connection = '{0}.s_t'.format(name)
        recurrent_connection_y = '{0}.y_t'.format(name)

        configs.append("# z_t")
        configs.append("component-node name={0}.z_t_pre component={0}.W_z.xs_z input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.z_t component={0}.z input={0}.z_t_pre".format(name))

        configs.append("# r_t")
        configs.append("component-node name={0}.r_t_pre component={0}.W_z.xs_r input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.r_t component={0}.r input={0}.r_t_pre".format(name))

        configs.append("# h_t")
        configs.append("component-node name={0}.h1_t component={0}.h1 input=Append({0}.r_t, IfDefined(Offset({1}, {2})))".format(name, recurrent_connection, delay))
        configs.append("component-node name={0}.h_t_pre component={0}.W_h.UW input=Append({1}, {0}.h1_t)".format(name, input_descriptor))
        configs.append("component-node name={0}.h_t component={0}.h input={0}.h_t_pre".format(name))

        configs.append("component-node name={0}.y1_t component={0}.y1 input=Append({0}.h_t, Sum(Scale(-1.0,{0}.z_t), Const(1.0, {1})))".format(name, cell_dim))
        configs.append("component-node name={0}.y2_t component={0}.y2 input=Append(IfDefined(Offset({1}, {2})), {0}.z_t)".format(name, recurrent_connection_y, delay))
        
        configs.append("component-node name={0}.y_t component={0}.y input=Sum({0}.y1_t, {0}.y2_t)".format(name))

        configs.append("# s_t recurrent")
        configs.append("component name={0}.W_s.ys type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, rec_proj_dim + nonrec_proj_dim, affine_str))
        configs.append("component name={0}.s_r type=BackpropTruncationComponent dim={1} {2}".format(name, rec_proj_dim, bptrunc_str))

        configs.append("# s_t and n_t : sn_t will be the output")
        configs.append("component-node name={0}.sn_t component={0}.W_s.ys input={0}.y_t".format(name))
        configs.append("dim-range-node name={0}.s_t_preclip input-node={0}.sn_t dim-offset=0 dim={1}".format(name, rec_proj_dim))
        configs.append("component-node name={0}.s_t component={0}.s_r input={0}.s_t_preclip".format(name))

        return configs


# This class is for lines like
#   'norm-pgru-layer name=norm-pgru1 input=[-1] delay=-3'

# Different from the vanilla PGRU, the NormPGRU uses batchnorm in the forward direction
# and renorm in the recurrence.

# The output dimension of the layer may be specified via 'cell-dim=xxx', but if not specified,
# the dimension defaults to the same as the input.
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   cell-dim=-1            [Dimension of the cell]
#   recurrent-projection-dim [Dimension of the projection used in recurrent connections, e.g. cell-dim/4]
#   non-recurrent-projection-dim   [Dimension of the projection in non-recurrent connections,
#                                   in addition to recurrent-projection-dim, e.g. cell-dim/4]
#   delay=-1                 [Delay in the recurrent connections of the GRU ]
#   clipping-threshold=30    [nnet3 GRU use a gradient clipping component at the recurrent connections.
#                             This is the threshold used to decide if clipping has to be activated ]
#   zeroing-interval=20      [interval at which we (possibly) zero out the recurrent derivatives.]
#   zeroing-threshold=15     [We only zero out the derivs every zeroing-interval, if derivs exceed this value.]
#   self-repair-scale-nonlinearity=1e-5      [It is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent]
#                                       i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent ]
#   ng-per-element-scale-options=''   [Additional options used for the diagonal matrices in the GRU ]
#   ng-affine-options=''              [Additional options used for the full matrices in the GRU, can be used to do things like set biases to initialize to 1]

class XconfigNormPgruLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "norm-pgru-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input' : '[-1]',
                        'cell-dim' : -1, # this is a compulsory argument
                        'recurrent-projection-dim' : -1,  # defaults to cell-dim / 4
                        'non-recurrent-projection-dim' : -1, # defaults to
                                                             # recurrent-projection-dim
                        'clipping-threshold' : 30.0,
                        'delay' : -1,
                        'ng-per-element-scale-options' : ' max-change=0.75 ',
                        'ng-affine-options' : ' max-change=0.75 ',
                        'self-repair-scale-nonlinearity' : 0.00001,
                        'zeroing-interval' : 20,
                        'zeroing-threshold' : 15.0,
                        'dropout-proportion' : -1.0, # If -1.0, no dropout components will be added
                        'dropout-per-frame' : True # If False, regular dropout, not per frame.
                       }

    def set_derived_configs(self):
        if self.config['recurrent-projection-dim'] <= 0:
            self.config['recurrent-projection-dim'] = self.config['cell-dim'] / 4

        if self.config['non-recurrent-projection-dim'] <= 0:
            self.config['non-recurrent-projection-dim'] = \
               self.config['recurrent-projection-dim']

    def check_configs(self):
        for key in ['cell-dim', 'recurrent-projection-dim',
                    'non-recurrent-projection-dim']:
            if self.config[key] <= 0:
                raise RuntimeError("{0} has invalid value {1}.".format(
                    key, self.config[key]))

        if self.config['delay'] == 0:
            raise RuntimeError("delay cannot be zero")

        if (self.config['recurrent-projection-dim'] +
            self.config['non-recurrent-projection-dim'] >
            self.config['cell-dim']):
            raise RuntimeError("recurrent+non-recurrent projection dim exceeds "
                                "cell dim.")
        for key in ['self-repair-scale-nonlinearity']:
            if self.config[key] < 0.0 or self.config[key] > 1.0:
                raise RuntimeError("{0} has invalid value {2}."
                                   .format(self.layer_type, key,
                                           self.config[key]))
        if ((self.config['dropout-proportion'] > 1.0 or
             self.config['dropout-proportion'] < 0.0) and
             self.config['dropout-proportion'] != -1.0 ):
             raise RuntimeError("dropout-proportion has invalid value {0}."
                                .format(self.config['dropout-proportion']))

    def auxiliary_outputs(self):
        return ['h_t']

    def output_name(self, auxiliary_output = None):
        node_name = 'sn_t'
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                node_name = auxiliary_output
            else:
                raise Exception("In {0} of type {1}, unknown auxiliary output name {1}".format(self.layer_type, auxiliary_output))

        return '{0}.{1}'.format(self.name, node_name)

    def output_dim(self, auxiliary_output = None):
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                if node_name == 'c_t':
                    return self.config['cell-dim']
                # add code for other auxiliary_outputs here when we decide to expose them
            else:
                raise Exception("In {0} of type {1}, unknown auxiliary output name {1}".format(self.layer_type, auxiliary_output))

        return self.config['recurrent-projection-dim'] + self.config['non-recurrent-projection-dim']

    def get_full_config(self):
        ans = []
        config_lines = self.generate_pgru_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in LSTM initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the PGRU config
    def generate_pgru_config(self):

        # assign some variables to reduce verbosity
        name = self.name
        # in the below code we will just call descriptor_strings as descriptors for conciseness
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        cell_dim = self.config['cell-dim']
        rec_proj_dim = self.config['recurrent-projection-dim']
        nonrec_proj_dim = self.config['non-recurrent-projection-dim']
        delay = self.config['delay']
        repair_nonlin = self.config['self-repair-scale-nonlinearity']
        repair_nonlin_str = "self-repair-scale={0:.10f}".format(repair_nonlin) if repair_nonlin is not None else ''
        bptrunc_str = ("clipping-threshold={0}"
                      " zeroing-threshold={1}"
                      " zeroing-interval={2}"
                      " recurrence-interval={3}"
                      "".format(self.config['clipping-threshold'],
                                self.config['zeroing-threshold'],
                                self.config['zeroing-interval'],
                                abs(delay)))
        affine_str = self.config['ng-affine-options']
        pes_str = self.config['ng-per-element-scale-options']
        dropout_proportion = self.config['dropout-proportion']
        dropout_per_frame = 'true' if self.config['dropout-per-frame'] else 'false' 

        # Natural gradient per element scale parameters
        # TODO: decide if we want to keep exposing these options
        if re.search('param-mean', pes_str) is None and \
           re.search('param-stddev', pes_str) is None:
           pes_str += " param-mean=0.0 param-stddev=1.0 "

        # formulation like:
        # z_t = \sigmoid ( x_t * U^z + s_{t-1} * W^z ) // update gate
        # r_t = \sigmoid ( x_t * U^r + s_{t-1} * W^r ) // reset gate
        # \tilde{h}_t = \tanh ( x_t * U^h + ( s_{t-1} \dot r_t ) * W^h )
        # h_t = ( 1 - z_t ) \dot \tilde{h}_t + z_t \dot h_{t-1}
        # y_t_tmp = h_t * W^y
        # s_t = renorm ( y_t_tmp (0:rec_proj_dim-1) )
        # y_t = batchnorm ( y_t_tmp )
        
        configs = []
        configs.append("# Update gate control : W_z* matrics")
        configs.append("component name={0}.W_z.xs_z type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, cell_dim, affine_str))
        
        configs.append("# Reset gate control : W_r* matrics")
        configs.append("component name={0}.W_z.xs_r type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, rec_proj_dim, affine_str))

        configs.append("# h related matrix : W_h* matrics")
        configs.append("component name={0}.W_h.UW type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, cell_dim , affine_str))
        
        if dropout_proportion != -1.0:
            configs.append("component name={0}.dropout_z type=DropoutComponent dim={1} "
                           "dropout-proportion={2} dropout-per-frame={3}"
                           .format(name, cell_dim, dropout_proportion, dropout_per_frame))
            configs.append("component name={0}.dropout_r type=DropoutComponent dim={1} "
                           "dropout-proportion={2} dropout-per-frame={3}"
                           .format(name, rec_proj_dim, dropout_proportion, dropout_per_frame))
        
        configs.append("# Defining the non-linearities")
        configs.append("component name={0}.z type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.r type=SigmoidComponent dim={1} {2}".format(name, rec_proj_dim, repair_nonlin_str))
        configs.append("component name={0}.h type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))

        configs.append("# Defining the components for other cell computations")
        configs.append("component name={0}.h1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * rec_proj_dim, rec_proj_dim))
        configs.append("component name={0}.y1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.y2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.y type=NoOpComponent dim={1}".format(name, cell_dim))

        recurrent_connection = '{0}.s_t'.format(name)
        recurrent_connection_y = '{0}.y_t'.format(name)

        configs.append("# z_t")
        configs.append("component-node name={0}.z_t_pre component={0}.W_z.xs_z input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        if dropout_proportion != -1.0:
            configs.append("component-node name={0}.z_predrop_t component={0}.z input={0}.z_t_pre".format(name))
            configs.append("component-node name={0}.z_t component={0}.dropout_z input={0}.z_predrop_t".format(name))
        else:
            configs.append("component-node name={0}.z_t component={0}.z input={0}.z_t_pre".format(name, input_descriptor, recurrent_connection, delay))

        configs.append("# r_t")
        configs.append("component-node name={0}.r_t_pre component={0}.W_z.xs_r input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        if dropout_proportion != -1.0:
            configs.append("component-node name={0}.r_predrop_t component={0}.r input={0}.r_t_pre".format(name))
            configs.append("component-node name={0}.r_t component={0}.dropout_r input={0}.r_predrop_t".format(name))            
        else:
            configs.append("component-node name={0}.r_t component={0}.r input={0}.r_t_pre".format(name))

        configs.append("# h_t")
        configs.append("component-node name={0}.h1_t component={0}.h1 input=Append({0}.r_t, IfDefined(Offset({1}, {2})))".format(name, recurrent_connection, delay))
        configs.append("component-node name={0}.h_t_pre component={0}.W_h.UW input=Append({1}, {0}.h1_t)".format(name, input_descriptor))
        configs.append("component-node name={0}.h_t component={0}.h input={0}.h_t_pre".format(name))

        configs.append("component-node name={0}.y1_t component={0}.y1 input=Append({0}.h_t, Sum(Scale(-1.0,{0}.z_t), Const(1.0, {1})))".format(name, cell_dim))
        configs.append("component-node name={0}.y2_t component={0}.y2 input=Append(IfDefined(Offset({1}, {2})), {0}.z_t)".format(name, recurrent_connection_y, delay))
        configs.append("component-node name={0}.y_t component={0}.y input=Sum({0}.y1_t, {0}.y2_t)".format(name))

        configs.append("# s_t recurrent")
        configs.append("component name={0}.W_s.ys type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, rec_proj_dim + nonrec_proj_dim, affine_str))
        configs.append("component name={0}.s_r type=BackpropTruncationComponent dim={1} {2}".format(name, rec_proj_dim, bptrunc_str))
        
        configs.append("component name={0}.batchnorm type=BatchNormComponent dim={1} target-rms=1.0".format(name, rec_proj_dim + nonrec_proj_dim))
        configs.append("component name={0}.renorm type=NormalizeComponent dim={1} target-rms=1.0".format(name, rec_proj_dim))

        configs.append("# s_t and n_t : sn_t will be the output")
        configs.append("component-node name={0}.sn_nobatchnorm_t component={0}.W_s.ys input={0}.y_t".format(name))
        configs.append("dim-range-node name={0}.s_t_preclip input-node={0}.sn_nobatchnorm_t dim-offset=0 dim={1}".format(name, rec_proj_dim))
        configs.append("component-node name={0}.sn_t component={0}.batchnorm input={0}.sn_nobatchnorm_t".format(name))

        configs.append("component-node name={0}.s_renorm_t component={0}.renorm input={0}.s_t_preclip".format(name))
        configs.append("component-node name={0}.s_t component={0}.s_r input={0}.s_renorm_t".format(name))

        return configs



# This class is for lines like
#   'opgru-layer name=opgru1 input=[-1] delay=-3'
# It generates an OPGRU sub-graph with output projections.
# The output dimension of the layer may be specified via 'cell-dim=xxx', but if not specified,
# the dimension defaults to the same as the input.
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   cell-dim=-1            [Dimension of the cell]
#   recurrent-projection-dim [Dimension of the projection used in recurrent connections, e.g. cell-dim/4]
#   non-recurrent-projection-dim   [Dimension of the projection in non-recurrent connections,
#                                   in addition to recurrent-projection-dim, e.g. cell-dim/4]
#   delay=-1                 [Delay in the recurrent connections of the GRU ]
#   clipping-threshold=30    [nnet3 GRU use a gradient clipping component at the recurrent connections.
#                             This is the threshold used to decide if clipping has to be activated ]
#   zeroing-interval=20      [interval at which we (possibly) zero out the recurrent derivatives.]
#   zeroing-threshold=15     [We only zero out the derivs every zeroing-interval, if derivs exceed this value.]
#   self-repair-scale-nonlinearity=1e-5      [It is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent]
#                                       i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent ]
#   ng-per-element-scale-options=''   [Additional options used for the diagonal matrices in the GRU ]
#   ng-affine-options=''              [Additional options used for the full matrices in the GRU, can be used to do things like set biases to initialize to 1]
class XconfigOpgruLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "opgru-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input' : '[-1]',
                        'cell-dim' : -1, # this is a compulsory argument
                        'recurrent-projection-dim' : -1,  # defaults to cell-dim / 4
                        'non-recurrent-projection-dim' : -1, # defaults to
                                                             # recurrent-projection-dim
                        'clipping-threshold' : 30.0,
                        'delay' : -1,
                        'ng-per-element-scale-options' : ' max-change=0.75 ',
                        'ng-affine-options' : ' max-change=0.75 ',
                        'self-repair-scale-nonlinearity' : 0.00001,
                        'zeroing-interval' : 20,
                        'zeroing-threshold' : 15.0
                       }

    def set_derived_configs(self):
        if self.config['recurrent-projection-dim'] <= 0:
            self.config['recurrent-projection-dim'] = self.config['cell-dim'] / 4

        if self.config['non-recurrent-projection-dim'] <= 0:
            self.config['non-recurrent-projection-dim'] = \
               self.config['recurrent-projection-dim']

    def check_configs(self):
        for key in ['cell-dim', 'recurrent-projection-dim',
                    'non-recurrent-projection-dim']:
            if self.config[key] <= 0:
                raise RuntimeError("{0} has invalid value {1}.".format(
                    key, self.config[key]))

        if self.config['delay'] == 0:
            raise RuntimeError("delay cannot be zero")

        if (self.config['recurrent-projection-dim'] +
            self.config['non-recurrent-projection-dim'] >
            self.config['cell-dim']):
            raise RuntimeError("recurrent+non-recurrent projection dim exceeds "
                                "cell dim.")
        for key in ['self-repair-scale-nonlinearity']:
            if self.config[key] < 0.0 or self.config[key] > 1.0:
                raise RuntimeError("{0} has invalid value {2}."
                                   .format(self.layer_type, key,
                                           self.config[key]))

    def auxiliary_outputs(self):
        return ['h_t']

    def output_name(self, auxiliary_output = None):
        node_name = 'sn_t'
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                node_name = auxiliary_output
            else:
                raise Exception("In {0} of type {1}, unknown auxiliary output name {1}".format(self.layer_type, auxiliary_output))

        return '{0}.{1}'.format(self.name, node_name)

    def output_dim(self, auxiliary_output = None):
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                if node_name == 'c_t':
                    return self.config['cell-dim']
                # add code for other auxiliary_outputs here when we decide to expose them
            else:
                raise Exception("In {0} of type {1}, unknown auxiliary output name {1}".format(self.layer_type, auxiliary_output))

        return self.config['recurrent-projection-dim'] + self.config['non-recurrent-projection-dim']

    def get_full_config(self):
        ans = []
        config_lines = self.generate_pgru_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in LSTM initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the PGRU config
    def generate_pgru_config(self):

        # assign some variables to reduce verbosity
        name = self.name
        # in the below code we will just call descriptor_strings as descriptors for conciseness
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        cell_dim = self.config['cell-dim']
        rec_proj_dim = self.config['recurrent-projection-dim']
        nonrec_proj_dim = self.config['non-recurrent-projection-dim']
        delay = self.config['delay']
        repair_nonlin = self.config['self-repair-scale-nonlinearity']
        repair_nonlin_str = "self-repair-scale={0:.10f}".format(repair_nonlin) if repair_nonlin is not None else ''
        bptrunc_str = ("clipping-threshold={0}"
                      " zeroing-threshold={1}"
                      " zeroing-interval={2}"
                      " recurrence-interval={3}"
                      "".format(self.config['clipping-threshold'],
                                self.config['zeroing-threshold'],
                                self.config['zeroing-interval'],
                                abs(delay)))
        affine_str = self.config['ng-affine-options']
        pes_str = self.config['ng-per-element-scale-options']

        # Natural gradient per element scale parameters
        # TODO: decide if we want to keep exposing these options
        if re.search('param-mean', pes_str) is None and \
           re.search('param-stddev', pes_str) is None:
           pes_str += " param-mean=0.0 param-stddev=1.0 "

        # formulation for OPGRU like:
        # z_t = \sigmoid ( x_t * U^z + s_{t-1} * W^z ) // update gate
        # o_t = \sigmoid ( x_t * U^o + s_{t-1} * W^o ) // output gate
        # \tilde{h}_t = \tanh ( x_t * U^h + h_{t-1} \dot W^h ) // W^h is learnable vector
        # h_t = ( 1 - z_t ) \dot \tilde{h}_t + z_t \dot h_{t-1}
        # y_t = (y_t \dot o_t) * W^y
        # s_t = y_t(0:rec_proj_dim-1)
        
        configs = []
        configs.append("# Update gate control : W_z* matrics")
        configs.append("component name={0}.W_z.xs_z type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, cell_dim, affine_str))
        
        configs.append("# Output gate control : W_r* matrics")
        configs.append("component name={0}.W_z.xs_o type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim + rec_proj_dim, cell_dim, affine_str))

        configs.append("# h related matrix : W_h* matrics")
        configs.append("component name={0}.W_h.UW type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, input_dim , cell_dim , affine_str))
        configs.append("component name={0}.W_h.UW_elementwise type=NaturalGradientPerElementScaleComponent dim={1} {2}".format(name, cell_dim , pes_str))
        
        configs.append("# Defining the non-linearities")
        configs.append("component name={0}.z type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.o type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.h type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))

        configs.append("# Defining the components for other cell computations")
        configs.append("component name={0}.o1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.y1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.y2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.y type=NoOpComponent dim={1}".format(name, cell_dim))

        recurrent_connection = '{0}.s_t'.format(name)
        recurrent_connection_y = '{0}.y_t'.format(name)

        configs.append("# z_t")
        configs.append("component-node name={0}.z_t_pre component={0}.W_z.xs_z input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.z_t component={0}.z input={0}.z_t_pre".format(name))

        configs.append("# o_t")
        configs.append("component-node name={0}.o_t_pre component={0}.W_z.xs_o input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        configs.append("component-node name={0}.o_t component={0}.o input={0}.o_t_pre".format(name))
        
        configs.append("# h_t")
        configs.append("component-node name={0}.h_t_pre component={0}.W_h.UW input={1}".format(name, input_descriptor))
        configs.append("component-node name={0}.h_t_pre2 component={0}.W_h.UW_elementwise input=IfDefined(Offset({1}, {2}))".format(name, recurrent_connection_y, delay))
        configs.append("component-node name={0}.h_t component={0}.h input=Sum({0}.h_t_pre, {0}.h_t_pre2)".format(name))

        configs.append("component-node name={0}.y1_t component={0}.y1 input=Append({0}.h_t, Sum(Scale(-1.0,{0}.z_t), Const(1.0, {1})))".format(name, cell_dim))
        configs.append("component-node name={0}.y2_t component={0}.y2 input=Append(IfDefined(Offset({1}, {2})), {0}.z_t)".format(name, recurrent_connection_y, delay))
        configs.append("component-node name={0}.y_t component={0}.y input=Sum({0}.y1_t, {0}.y2_t)".format(name))
        configs.append("component-node name={0}.y_o_t component={0}.o1 input=Append({0}.o_t, {0}.y_t)".format(name))

        configs.append("# s_t recurrent")
        configs.append("component name={0}.W_s.ys type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3}".format(name, cell_dim, rec_proj_dim + nonrec_proj_dim, affine_str))
        configs.append("component name={0}.s_r type=BackpropTruncationComponent dim={1} {2}".format(name, rec_proj_dim, bptrunc_str))

        configs.append("# s_t and n_t : sn_t will be the output")
        configs.append("component-node name={0}.sn_t component={0}.W_s.ys input={0}.y_o_t".format(name))
        configs.append("dim-range-node name={0}.s_t_preclip input-node={0}.sn_t dim-offset=0 dim={1}".format(name, rec_proj_dim))
        configs.append("component-node name={0}.s_t component={0}.s_r input={0}.s_t_preclip".format(name))

        return configs

# This class is for lines like
#   'norm-opgru-layer name=norm-opgru1 input=[-1] delay=-3'
# It generates a norm-OPGRU sub-graph with output projections.

# Different from the vanilla OPGRU, the NormOPGRU uses batchnorm in the forward direction
# and renorm in the recurrence.

# The output dimension of the layer may be specified via 'cell-dim=xxx', but if not specified,
# the dimension defaults to the same as the input.
# See other configuration values below.
#
# Parameters of the class, and their defaults:
#   input='[-1]'             [Descriptor giving the input of the layer.]
#   cell-dim=-1            [Dimension of the cell]
#   recurrent-projection-dim [Dimension of the projection used in recurrent connections, e.g. cell-dim/4]
#   non-recurrent-projection-dim   [Dimension of the projection in non-recurrent connections,
#                                   in addition to recurrent-projection-dim, e.g. cell-dim/4]
#   delay=-1                 [Delay in the recurrent connections of the GRU ]
#   clipping-threshold=30    [nnet3 GRU use a gradient clipping component at the recurrent connections.
#                             This is the threshold used to decide if clipping has to be activated ]
#   zeroing-interval=20      [interval at which we (possibly) zero out the recurrent derivatives.]
#   zeroing-threshold=15     [We only zero out the derivs every zeroing-interval, if derivs exceed this value.]
#   self-repair-scale-nonlinearity=1e-5      [It is a constant scaling the self-repair vector computed in derived classes of NonlinearComponent]
#                                       i.e.,  SigmoidComponent, TanhComponent and RectifiedLinearComponent ]
#   ng-per-element-scale-options=''   [Additional options used for the diagonal matrices in the GRU ]
#   ng-affine-options=''              [Additional options used for the full matrices in the GRU, can be used to do things like set biases to initialize to 1]
class XconfigNormOpgruLayer(XconfigLayerBase):
    def __init__(self, first_token, key_to_value, prev_names = None):
        assert first_token == "norm-opgru-layer"
        XconfigLayerBase.__init__(self, first_token, key_to_value, prev_names)

    def set_default_configs(self):
        self.config = {'input' : '[-1]',
                        'cell-dim' : -1, # this is a compulsory argument
                        'recurrent-projection-dim' : -1,  # defaults to cell-dim / 4
                        'non-recurrent-projection-dim' : -1, # defaults to
                                                             # recurrent-projection-dim
                        'clipping-threshold' : 30.0,
                        'delay' : -1,
                        'ng-per-element-scale-options' : ' max-change=0.75 ',
                        'ng-affine-options' : ' max-change=0.75 ',
                        'self-repair-scale-nonlinearity' : 0.00001,
                        'zeroing-interval' : 20,
                        'zeroing-threshold' : 15.0,
                        'dropout-proportion' : -1.0, # If -1.0, no dropout components will be added
                        'l2-regularize': 0.0,
                        'dropout-per-frame' : True  # If false, regular dropout, not per frame.
                       }

    def set_derived_configs(self):
        if self.config['recurrent-projection-dim'] <= 0:
            self.config['recurrent-projection-dim'] = self.config['cell-dim'] / 4

        if self.config['non-recurrent-projection-dim'] <= 0:
            self.config['non-recurrent-projection-dim'] = \
               self.config['recurrent-projection-dim']

    def check_configs(self):
        for key in ['cell-dim', 'recurrent-projection-dim',
                    'non-recurrent-projection-dim']:
            if self.config[key] <= 0:
                raise RuntimeError("{0} has invalid value {1}.".format(
                    key, self.config[key]))

        if self.config['delay'] == 0:
            raise RuntimeError("delay cannot be zero")

        if (self.config['recurrent-projection-dim'] +
            self.config['non-recurrent-projection-dim'] >
            self.config['cell-dim']):
            raise RuntimeError("recurrent+non-recurrent projection dim exceeds "
                                "cell dim.")
        for key in ['self-repair-scale-nonlinearity']:
            if self.config[key] < 0.0 or self.config[key] > 1.0:
                raise RuntimeError("{0} has invalid value {2}."
                                   .format(self.layer_type, key,
                                           self.config[key]))
        if ((self.config['dropout-proportion'] > 1.0 or
             self.config['dropout-proportion'] < 0.0) and
             self.config['dropout-proportion'] != -1.0 ):
             raise RuntimeError("dropout-proportion has invalid value {0}."
                                .format(self.config['dropout-proportion']))

    def auxiliary_outputs(self):
        return ['h_t']

    def output_name(self, auxiliary_output = None):
        node_name = 'sn_t'
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                node_name = auxiliary_output
            else:
                raise Exception("In {0} of type {1}, unknown auxiliary output name {1}".format(self.layer_type, auxiliary_output))

        return '{0}.{1}'.format(self.name, node_name)

    def output_dim(self, auxiliary_output = None):
        if auxiliary_output is not None:
            if auxiliary_output in self.auxiliary_outputs():
                if node_name == 'c_t':
                    return self.config['cell-dim']
                # add code for other auxiliary_outputs here when we decide to expose them
            else:
                raise Exception("In {0} of type {1}, unknown auxiliary output name {1}".format(self.layer_type, auxiliary_output))

        return self.config['recurrent-projection-dim'] + self.config['non-recurrent-projection-dim']

    def get_full_config(self):
        ans = []
        config_lines = self.generate_pgru_config()

        for line in config_lines:
            for config_name in ['ref', 'final']:
                # we do not support user specified matrices in LSTM initialization
                # so 'ref' and 'final' configs are the same.
                ans.append((config_name, line))
        return ans

    # convenience function to generate the PGRU config
    def generate_pgru_config(self):

        # assign some variables to reduce verbosity
        name = self.name
        # in the below code we will just call descriptor_strings as descriptors for conciseness
        input_dim = self.descriptors['input']['dim']
        input_descriptor = self.descriptors['input']['final-string']
        cell_dim = self.config['cell-dim']
        rec_proj_dim = self.config['recurrent-projection-dim']
        nonrec_proj_dim = self.config['non-recurrent-projection-dim']
        delay = self.config['delay']
        repair_nonlin = self.config['self-repair-scale-nonlinearity']
        repair_nonlin_str = "self-repair-scale={0:.10f}".format(repair_nonlin) if repair_nonlin is not None else ''
        bptrunc_str = ("clipping-threshold={0}"
                      " zeroing-threshold={1}"
                      " zeroing-interval={2}"
                      " recurrence-interval={3}"
                      "".format(self.config['clipping-threshold'],
                                self.config['zeroing-threshold'],
                                self.config['zeroing-interval'],
                                abs(delay)))
        affine_str = self.config['ng-affine-options']
        pes_str = self.config['ng-per-element-scale-options']
        dropout_proportion = self.config['dropout-proportion']
        dropout_per_frame = 'true' if self.config['dropout-per-frame'] else 'false' 

        l2_regularize = self.config['l2-regularize']
        l2_regularize_option = ('l2-regularize={0} '.format(l2_regularize)
                                if l2_regularize != 0.0 else '')

        # Natural gradient per element scale parameters
        # TODO: decide if we want to keep exposing these options
        if re.search('param-mean', pes_str) is None and \
           re.search('param-stddev', pes_str) is None:
           pes_str += " param-mean=0.0 param-stddev=1.0 "

        # formulation for OPGRU like:
        # z_t = \sigmoid ( x_t * U^z + s_{t-1} * W^z ) // update gate
        # o_t = \sigmoid ( x_t * U^o + s_{t-1} * W^o ) // output gate
        # \tilde{h}_t = \tanh ( x_t * U^h + h_{t-1} \dot W^h ) // W^h is learnable vector
        # h_t = ( 1 - z_t ) \dot \tilde{h}_t + z_t \dot h_{t-1}
        # y_t_tmp = ( h_t \dot o_t) * W^y
        # s_t = renorm ( y_t_tmp(0:rec_proj_dim-1) )
        # y_t = batchnorm ( y_t_tmp )
        
        configs = []
        configs.append("# Update gate control : W_z* matrics")
        configs.append("component name={0}.W_z.xs_z type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3} {4}".format(name, input_dim + rec_proj_dim, cell_dim, affine_str, l2_regularize_option))
        
        configs.append("# Output gate control : W_r* matrics")
        configs.append("component name={0}.W_z.xs_o type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3} {4}".format(name, input_dim + rec_proj_dim, cell_dim, affine_str, l2_regularize_option))

        configs.append("# h related matrix : W_h* matrics")
        configs.append("component name={0}.W_h.UW type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3} {4}".format(name, input_dim , cell_dim , affine_str, l2_regularize_option))
        configs.append("component name={0}.W_h.UW_elementwise type=NaturalGradientPerElementScaleComponent dim={1} {2}".format(name, cell_dim , pes_str))
        
        configs.append("# Defining the non-linearities")
        configs.append("component name={0}.z type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.o type=SigmoidComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))
        configs.append("component name={0}.h type=TanhComponent dim={1} {2}".format(name, cell_dim, repair_nonlin_str))

        configs.append("# Defining the components for other cell computations")
        configs.append("component name={0}.o1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.y1 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.y2 type=ElementwiseProductComponent input-dim={1} output-dim={2}".format(name, 2 * cell_dim, cell_dim))
        configs.append("component name={0}.y type=NoOpComponent dim={1}".format(name, cell_dim))

        if dropout_proportion != -1.0:
            configs.append("component name={0}.dropout type=DropoutComponent dim={1} "
                           "dropout-proportion={2} dropout-per-frame={3}"
                           .format(name, cell_dim, dropout_proportion, dropout_per_frame))

        recurrent_connection = '{0}.s_t'.format(name)
        recurrent_connection_y = '{0}.y_t'.format(name)

        configs.append("# z_t")
        configs.append("component-node name={0}.z_t_pre component={0}.W_z.xs_z input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        if dropout_proportion != -1.0:
            configs.append("component-node name={0}.z_predrop_t component={0}.z input={0}.z_t_pre".format(name))
            configs.append("component-node name={0}.z_t component={0}.dropout input={0}.z_predrop_t".format(name))
        else:
            configs.append("component-node name={0}.z_t component={0}.z input={0}.z_t_pre".format(name))

        configs.append("# o_t")
        configs.append("component-node name={0}.o_t_pre component={0}.W_z.xs_o input=Append({1}, IfDefined(Offset({2}, {3})))".format(name, input_descriptor, recurrent_connection, delay))
        if dropout_proportion != -1.0:
            configs.append("component-node name={0}.o_predrop_t component={0}.o input={0}.o_t_pre".format(name))
            configs.append("component-node name={0}.o_t component={0}.dropout input={0}.o_predrop_t".format(name))
        else:
            configs.append("component-node name={0}.o_t component={0}.o input={0}.o_t_pre".format(name))
        
        configs.append("# h_t")
        configs.append("component-node name={0}.h_t_pre component={0}.W_h.UW input={1}".format(name, input_descriptor))
        configs.append("component-node name={0}.h_t_pre2 component={0}.W_h.UW_elementwise input=IfDefined(Offset({1}, {2}))".format(name, recurrent_connection_y, delay))
        configs.append("component-node name={0}.h_t component={0}.h input=Sum({0}.h_t_pre, {0}.h_t_pre2)".format(name))

        configs.append("# The following two lines are to implement (1 - z_t)")
        configs.append("component-node name={0}.y1_t component={0}.y1 input=Append({0}.h_t, Sum(Scale(-1.0,{0}.z_t), Const(1.0, {1})))".format(name, cell_dim))
        configs.append("component-node name={0}.y2_t component={0}.y2 input=Append(IfDefined(Offset({1}, {2})), {0}.z_t)".format(name, recurrent_connection_y, delay))
        configs.append("component-node name={0}.y_t component={0}.y input=Sum({0}.y1_t, {0}.y2_t)".format(name))
        configs.append("component-node name={0}.y_o_t component={0}.o1 input=Append({0}.o_t, {0}.y_t)".format(name))

        configs.append("# s_t recurrent")
        configs.append("component name={0}.W_s.ys type=NaturalGradientAffineComponent input-dim={1} output-dim={2} {3} {4}".format(name, cell_dim, rec_proj_dim + nonrec_proj_dim, affine_str, l2_regularize_option))
        configs.append("component name={0}.s_r type=BackpropTruncationComponent dim={1} {2}".format(name, rec_proj_dim, bptrunc_str))
        configs.append("component name={0}.batchnorm type=BatchNormComponent dim={1} target-rms=1.0".format(name, rec_proj_dim + nonrec_proj_dim))
        configs.append("component name={0}.renorm type=NormalizeComponent dim={1} target-rms=1.0".format(name, rec_proj_dim))

        configs.append("# s_t and n_t : sn_t will be the output")
        configs.append("component-node name={0}.sn_nobatchnorm_t component={0}.W_s.ys input={0}.y_o_t".format(name))
        configs.append("component-node name={0}.sn_t component={0}.batchnorm input={0}.sn_nobatchnorm_t".format(name))
        configs.append("dim-range-node name={0}.s_t_preclip input-node={0}.sn_nobatchnorm_t dim-offset=0 dim={1}".format(name, rec_proj_dim))
        configs.append("component-node name={0}.s_t_preclip_renorm component={0}.renorm input={0}.s_t_preclip".format(name))
        configs.append("component-node name={0}.s_t component={0}.s_r input={0}.s_t_preclip_renorm".format(name))

        return configs
