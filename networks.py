# The implementation of GDN is inherited from
# https://github.com/tensorflow/compression,
# under the Apache License, Version 2.0. The 
# source code also include an implementation
# of the arithmetic coding by Nayuki from
# https://github.com/nayuki/Reference-arithmetic-coding
# under the MIT License.
#
# This file is being made available under the BSD License.  
# Copyright (c) 2020 Yueyu Hu

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import glob

import arithmeticcoding

import numpy as np
import tensorflow as tf
import pickle

from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.python.framework import tensor_shape

import math
import time

@tf.custom_gradient
def bypass_round(x):
  def _by_pass_grad(dy):
    return dy
  return tf.round(x), _by_pass_grad

@tf.RegisterGradient("UpperBound")
def _upper_bound_grad(op, grad):
  inputs, bound = op.inputs
  pass_through_if = tf.math.logical_or(inputs <= bound, grad > 0)
  return [tf.cast(pass_through_if, grad.dtype) * grad, None]


@tf.RegisterGradient("LowerBound")
def _lower_bound_grad(op, grad):
  inputs, bound = op.inputs
  pass_through_if = tf.math.logical_or(inputs >= bound, grad < 0)
  return [tf.cast(pass_through_if, grad.dtype) * grad, None]


def upper_bound(inputs, bound, gradient="identity_if_towards", name=None):
  try:
    gradient = {
        "identity_if_towards": "UpperBound",
        "identity": "IdentityFirstOfTwoInputs",
        "disconnected": None,
    }[gradient]
  except KeyError:
    raise ValueError("Invalid value for `gradient`: '{}'.".format(gradient))

  with tf.name_scope(name, "UpperBound", [inputs, bound]) as scope:
    inputs = tf.convert_to_tensor(inputs, name="inputs")
    bound = tf.convert_to_tensor(
        bound, name="bound", dtype=inputs.dtype)
    if gradient:
      with tf.get_default_graph().gradient_override_map({"Minimum": gradient}):
        return tf.math.minimum(inputs, bound, name=scope)
    else:
      return tf.math.minimum(inputs, bound, name=scope)

def lower_bound(inputs, bound, gradient="identity_if_towards", name=None):
  try:
    gradient = {
        "identity_if_towards": "LowerBound",
        "identity": "IdentityFirstOfTwoInputs",
        "disconnected": None,
    }[gradient]
  except KeyError:
    raise ValueError("Invalid value for `gradient`: '{}'.".format(gradient))

  with tf.name_scope(name, "LowerBound", [inputs, bound]) as scope:
    inputs = tf.convert_to_tensor(inputs, name="inputs")
    bound = tf.convert_to_tensor(
        bound, name="bound", dtype=inputs.dtype)
    if gradient:
      with tf.get_default_graph().gradient_override_map({"Maximum": gradient}):
        return tf.math.maximum(inputs, bound, name=scope)
    else:
      return tf.math.maximum(inputs, bound, name=scope)


def load_image(filename):
  """Loads a PNG image file."""

  string = tf.read_file(filename)
  image = tf.image.decode_image(string, channels=3)
  image = tf.cast(image, tf.float32)
  image /= 127.5
  image -= 1.0
  return image

def quantize_image(image):
  image = tf.clip_by_value(image, -1, 1)
  image += 1
  image = tf.round(image * 127.5)
  image = tf.cast(image, tf.uint8)
  return image

def save_image(filename, image):
  """Saves an image to a PNG file."""
  image = quantize_image(image)
  string = tf.image.encode_png(image)
  return tf.write_file(filename, string)

class DeterminedParameterizer(object):
  def __init__(self, minimum=0, reparam_offset=2 ** -18):
    self.minimum = float(minimum)
    self.reparam_offset = float(reparam_offset)

  def __call__(self, getter, name, shape, dtype, initializer, regularizer=None):
    pedestal = tf.constant(self.reparam_offset ** 2, dtype=dtype)
    bound = tf.constant(
        (self.minimum + self.reparam_offset ** 2) ** .5, dtype=dtype)
    reparam_name = "reparam_" + name

    def reparam_initializer(shape, dtype=None, partition_info=None):
      init = initializer(shape, dtype=dtype, partition_info=partition_info)
      return init

    def reparam(var):
      var = lower_bound(var, bound)
      var = tf.square(var) - pedestal
      return var

    if regularizer is not None:
      regularizer = lambda rdft: regularizer(reparam(rdft))

    var = getter(
        name=reparam_name, shape=shape, dtype=dtype,
        initializer=reparam_initializer, regularizer=regularizer)
    return reparam(var)

class NonnegativeParameterizer(object):
  def __init__(self, minimum=0, reparam_offset=2 ** -18):
    self.minimum = float(minimum)
    self.reparam_offset = float(reparam_offset)

  def __call__(self, getter, name, shape, dtype, initializer, regularizer=None):
    pedestal = tf.constant(self.reparam_offset ** 2, dtype=dtype)
    bound = tf.constant(
        (self.minimum + self.reparam_offset ** 2) ** .5, dtype=dtype)
    reparam_name = "reparam_" + name

    def reparam_initializer(shape, dtype=None, partition_info=None):
      init = initializer(shape, dtype=dtype, partition_info=partition_info)
      init = tf.math.sqrt(init + pedestal)
      return init

    def reparam(var):
      var = lower_bound(var, bound)
      var = tf.math.square(var) - pedestal
      return var

    if regularizer is not None:
      regularizer = lambda rdft: regularizer(reparam(rdft))

    var = getter(
        name=reparam_name, shape=shape, dtype=dtype,
        initializer=reparam_initializer, regularizer=regularizer)
    return reparam(var)

class LoadInitializer():
  def __init__(self, pk_fn):
    super(LoadInitializer, self).__init__()
    self.pk_fn = pk_fn
    with open(pk_fn, 'rb') as f:
      self.weight_dict = pickle.load(f)

  def init(self, name):
    def f(*args, **kwargs):
      d = np.array(self.weight_dict[name])
      return d
    return f

_default_beta_param = NonnegativeParameterizer(minimum=1e-6)
_default_gamma_param = NonnegativeParameterizer()

class GDN(tf.keras.layers.Layer):
  def __init__(self,
               inverse=False,
               rectify=False,
               gamma_init=.1,
               data_format="channels_last",
               beta_parameterizer=_default_beta_param,
               gamma_parameterizer=_default_gamma_param,
               beta_initializer=tf.initializers.ones(),
               gamma_initializer=tf.initializers.identity(gain=.1),
               activity_regularizer=None,
               trainable=True,
               name=None,
               **kwargs):
    super(GDN, self).__init__(trainable=trainable, name=name,
                              activity_regularizer=activity_regularizer,
                              **kwargs)
    self.inverse = bool(inverse)
    self.rectify = bool(rectify)
    self._gamma_init = float(gamma_init)
    self.data_format = data_format
    self._beta_parameterizer = beta_parameterizer
    self._gamma_parameterizer = gamma_parameterizer
    self._channel_axis()  # trigger ValueError early
    self.input_spec = tf.keras.layers.InputSpec(min_ndim=2)
    self.gamma_initializer = gamma_initializer
    self.beta_initializer = beta_initializer

  def _channel_axis(self):
    try:
      return {"channels_first": 1, "channels_last": -1}[self.data_format]
    except KeyError:
      raise ValueError("Unsupported `data_format` for GDN layer: {}.".format(
          self.data_format))

  def build(self, input_shape):
    channel_axis = self._channel_axis()
    input_shape = tensor_shape.TensorShape(input_shape)
    num_channels = input_shape[channel_axis].value
    if num_channels is None:
      raise ValueError("The channel dimension of the inputs to `GDN` "
                       "must be defined.")
    self._input_rank = input_shape.ndims
    self.input_spec = tf.keras.layers.InputSpec(ndim=input_shape.ndims,
                                     axes={channel_axis: num_channels})

    self.beta = self._beta_parameterizer(
        name="beta", shape=[num_channels], dtype=self.dtype,
        getter=self.add_variable, initializer=self.beta_initializer)

    self.gamma = self._gamma_parameterizer(
        name="gamma", shape=[num_channels, num_channels], dtype=self.dtype,
        getter=self.add_variable,
        initializer=self.gamma_initializer)

    self.built = True

  def call(self, inputs):
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    ndim = self._input_rank

    if self.rectify:
      inputs = tf.nn.relu(inputs)

    # Compute normalization pool.
    if ndim == 2:
      norm_pool = tf.math.matmul(tf.math.square(inputs), self.gamma)
      norm_pool = tf.nn.bias_add(norm_pool, self.beta)
    elif self.data_format == "channels_last" and ndim <= 5:
      shape = self.gamma.shape.as_list()
      gamma = tf.reshape(self.gamma, (ndim - 2) * [1] + shape)
      norm_pool = tf.nn.convolution(tf.math.square(inputs), gamma, "VALID")
      norm_pool = tf.nn.bias_add(norm_pool, self.beta)
    else:  # generic implementation
      # This puts channels in the last dimension regardless of input.
      norm_pool = tf.math.tensordot(
          tf.math.square(inputs), self.gamma, [[self._channel_axis()], [0]])
      norm_pool += self.beta
      if self.data_format == "channels_first":
        # Return to channels_first format if necessary.
        axes = list(range(ndim - 1))
        axes.insert(1, ndim - 1)
        norm_pool = tf.transpose(norm_pool, axes)

    if self.inverse:
      norm_pool = tf.sqrt(norm_pool)
    else:
      norm_pool = tf.rsqrt(norm_pool)
    outputs = inputs * norm_pool

    return outputs

  def compute_output_shape(self, input_shape):
    return tensor_shape.TensorShape(input_shape)

class analysisTransformModel(tf.keras.Model):
  def __init__(self, num_filters, conv_trainable=True, gdn_trainable=True, suffix='', init_loader=None):
    super(analysisTransformModel, self).__init__()
    self.layer_1 = Conv2D(
          num_filters, (5, 5), strides=(2, 2), name='layre_1%s' % (suffix), padding="SAME", kernel_initializer=init_loader.init('RE6_GPU0/analysis_transform_model/layre_1_mainA/kernel'),
          bias_initializer=init_loader.init('RE6_GPU0/analysis_transform_model/layre_1_mainA/bias'),
          use_bias=True, activation=GDN(beta_parameterizer=DeterminedParameterizer(), gamma_parameterizer=DeterminedParameterizer(),
          trainable=gdn_trainable, gamma_initializer=init_loader.init('RE6_GPU0/analysis_transform_model/layre_1_mainA/gdn/reparam_gamma'), 
          beta_initializer=init_loader.init('RE6_GPU0/analysis_transform_model/layre_1_mainA/gdn/reparam_beta')), trainable=conv_trainable)
    self.layer_2 = Conv2D(
          num_filters, (5, 5), strides=(2, 2), name='layre_2%s' % (suffix), padding="SAME", kernel_initializer=init_loader.init('RE6_GPU0/analysis_transform_model/layre_2_mainA/kernel'),
          bias_initializer=init_loader.init('RE6_GPU0/analysis_transform_model/layre_2_mainA/bias'),
          use_bias=True, activation=GDN(beta_parameterizer=DeterminedParameterizer(), gamma_parameterizer=DeterminedParameterizer(),
          trainable=gdn_trainable, gamma_initializer=init_loader.init('RE6_GPU0/analysis_transform_model/layre_2_mainA/gdn_1/reparam_gamma'), 
          beta_initializer=init_loader.init('RE6_GPU0/analysis_transform_model/layre_2_mainA/gdn_1/reparam_beta')), trainable=conv_trainable)
    self.layer_3 = Conv2D(
          num_filters, (5, 5), strides=(2, 2), name='layre_3%s' % (suffix), padding="SAME", kernel_initializer=init_loader.init('RE6_GPU0/analysis_transform_model/layre_3_mainA/kernel'),
          bias_initializer=init_loader.init('RE6_GPU0/analysis_transform_model/layre_3_mainA/bias'),
          use_bias=True, activation=GDN(beta_parameterizer=DeterminedParameterizer(), gamma_parameterizer=DeterminedParameterizer(),
          trainable=gdn_trainable, gamma_initializer=init_loader.init('RE6_GPU0/analysis_transform_model/layre_3_mainA/gdn_2/reparam_gamma'), 
          beta_initializer=init_loader.init('RE6_GPU0/analysis_transform_model/layre_3_mainA/gdn_2/reparam_beta')), trainable=conv_trainable)
    self.layer_4 = Conv2D(
          num_filters, (5, 5), strides=(2, 2), name='layre_4%s' % (suffix), padding="SAME", kernel_initializer=init_loader.init('RE6_GPU0/analysis_transform_model/layre_4_mainA/kernel'),
          bias_initializer=init_loader.init('RE6_GPU0/analysis_transform_model/layre_4_mainA/bias'),
          use_bias=True, activation=None, trainable=conv_trainable)

  def call(self, inputs):
    x = self.layer_1(inputs)
    x = self.layer_2(x)
    x = self.layer_3(x)
    x = self.layer_4(x)
    return x

class synthesisTransformModel(tf.keras.Model):
  def __init__(self, num_filters, conv_trainable=True, gdn_trainable=True, suffix='', init_loader=None):
    super(synthesisTransformModel, self).__init__()
    self.layer_1 = Conv2DTranspose(
        num_filters, (5, 5), strides=(2,2), name='layer_1%s' % (suffix), padding="SAME", kernel_initializer=init_loader.init('RE6_GPU0/synthesis_transform_model/layer_1_mainS/kernel'),
        bias_initializer=init_loader.init('RE6_GPU0/synthesis_transform_model/layer_1_mainS/bias'),
        use_bias=True, activation=GDN(beta_parameterizer=DeterminedParameterizer(), gamma_parameterizer=DeterminedParameterizer(),
        trainable=gdn_trainable, inverse=True, gamma_initializer=init_loader.init('RE6_GPU0/synthesis_transform_model/layer_1_mainS/gdn_3/reparam_gamma'), 
        beta_initializer=init_loader.init('RE6_GPU0/synthesis_transform_model/layer_1_mainS/gdn_3/reparam_beta')), trainable=conv_trainable)

    self.layer_2 = Conv2DTranspose(
        num_filters, (5, 5), strides=(2,2), name='layer_2%s' % (suffix), padding="SAME", kernel_initializer=init_loader.init('RE6_GPU0/synthesis_transform_model/layer_2_mainS/kernel'),
        bias_initializer=init_loader.init('RE6_GPU0/synthesis_transform_model/layer_2_mainS/bias'),
        use_bias=True, activation=GDN(beta_parameterizer=DeterminedParameterizer(), gamma_parameterizer=DeterminedParameterizer(),
        trainable=gdn_trainable, inverse=True, gamma_initializer=init_loader.init('RE6_GPU0/synthesis_transform_model/layer_2_mainS/gdn_4/reparam_gamma'), 
        beta_initializer=init_loader.init('RE6_GPU0/synthesis_transform_model/layer_2_mainS/gdn_4/reparam_beta')), trainable=conv_trainable)

    self.layer_3 = Conv2DTranspose(
        num_filters, (5, 5), strides=(2,2), name='layer_3%s' % (suffix), padding="SAME", kernel_initializer=init_loader.init('RE6_GPU0/synthesis_transform_model/layer_3_mainS/kernel'),
        bias_initializer=init_loader.init('RE6_GPU0/synthesis_transform_model/layer_3_mainS/bias'),
        use_bias=True, activation=GDN(beta_parameterizer=DeterminedParameterizer(), gamma_parameterizer=DeterminedParameterizer(),
        trainable=gdn_trainable, inverse=True, gamma_initializer=init_loader.init('RE6_GPU0/synthesis_transform_model/layer_3_mainS/gdn_5/reparam_gamma'), 
        beta_initializer=init_loader.init('RE6_GPU0/synthesis_transform_model/layer_3_mainS/gdn_5/reparam_beta')), trainable=conv_trainable)

    self.layer_4 = Conv2DTranspose(
        3, (5, 5), strides=(2,2), name='layer_4%s' % (suffix), padding="SAME", kernel_initializer=init_loader.init('RE6_GPU0/synthesis_transform_model/layer_4_mainS/kernel'),
        bias_initializer=init_loader.init('RE6_GPU0/synthesis_transform_model/layer_4_mainS/bias'),
        use_bias=True, activation=None, trainable=conv_trainable)

  def call(self, inputs):
    x = self.layer_1(inputs)
    x = self.layer_2(x)
    x = self.layer_3(x)
    pf = x
    x = self.layer_4(x)
    return x, pf

class h_analysisTransformModelLoad(tf.keras.Model):
  def __init__(self, num_filters, strides_list, conv_trainable=True, skip_level=-1, suffix='', prefix='', init_loader=None):
    super(h_analysisTransformModelLoad, self).__init__()
    self.layer_1 = Conv2D(num_filters[0], (3, 3), strides=strides_list[0], padding="SAME", name='layer_1%s' % (suffix),
                          kernel_initializer=init_loader.init(prefix+'layer_1'+suffix+'/kernel'),
                          bias_initializer=init_loader.init(prefix+'layer_1'+suffix+'/bias'),
                          use_bias=True, activation=None, trainable=conv_trainable)

    self.layer_2 = Conv2D(num_filters[1], (1, 1), strides=strides_list[1], padding="SAME", name='layer_2%s' % (suffix),
                          kernel_initializer=init_loader.init(prefix+'layer_2'+suffix+'/kernel'),
                          bias_initializer=init_loader.init(prefix+'layer_2'+suffix+'/bias'),
                          use_bias=True, activation=tf.nn.relu, trainable=conv_trainable)

    self.layer_3 = Conv2D(num_filters[1], (1, 1), strides=(1,1), padding="SAME", name='layer_3%s' % (suffix),
                          kernel_initializer=init_loader.init(prefix+'layer_3'+suffix+'/kernel'),
                          bias_initializer=init_loader.init(prefix+'layer_3'+suffix+'/bias'),
                          use_bias=True, activation=tf.nn.relu, trainable=conv_trainable)

    self.layer_4 = Conv2D(num_filters[2], (1, 1), strides=strides_list[2], padding="SAME", name='layer_4%s' % (suffix),
                          kernel_initializer=init_loader.init(prefix+'layer_4'+suffix+'/kernel'),
                          bias_initializer=init_loader.init(prefix+'layer_4'+suffix+'/bias'),
                          use_bias=True, activation=None, trainable=conv_trainable)

    self.skip_level = skip_level

  def call(self, inputs):
    x1 = self.layer_1(inputs)
    x1 = tf.space_to_depth(x1, 2)
    x2 = self.layer_2(x1)
    x3 = self.layer_3(x2)
    x = self.layer_4(x3)
    return x

class h_synthesisTransformModelLoad(tf.keras.Model):
  def __init__(self, num_filters, strides_list, conv_trainable=True, skip_level=-1, suffix='', prefix='', init_loader=None):
    super(h_synthesisTransformModelLoad, self).__init__()
    self.layer_1 = Conv2DTranspose(num_filters[0], (1, 1), strides=strides_list[2], padding="SAME", name='layer_1%s' % (suffix),
                          kernel_initializer=init_loader.init(prefix+'layer_1'+suffix+'/kernel'),
                          bias_initializer=init_loader.init(prefix+'layer_1'+suffix+'/bias'),
                          use_bias=True, activation=None,trainable=conv_trainable)

    self.layer_2 = Conv2DTranspose(num_filters[1], (1, 1), strides=strides_list[1], padding="SAME", name='layer_2%s' % (suffix),
                          kernel_initializer=init_loader.init(prefix+'layer_2'+suffix+'/kernel'),
                          bias_initializer=init_loader.init(prefix+'layer_2'+suffix+'/bias'),
                          use_bias=True, activation=tf.nn.relu, trainable=conv_trainable)

    self.layer_3 = Conv2DTranspose(num_filters[1], (1, 1), strides=(1,1), padding="SAME", name='layer_3%s' % (suffix),
                          kernel_initializer=init_loader.init(prefix+'layer_3'+suffix+'/kernel'),
                          bias_initializer=init_loader.init(prefix+'layer_3'+suffix+'/bias'),
                          use_bias=True, activation=tf.nn.relu, trainable=conv_trainable)

    self.layer_4 = Conv2DTranspose(num_filters[2], (3, 3), strides=strides_list[0], padding="SAME", name='layer_4%s' % (suffix),
                          kernel_initializer=init_loader.init(prefix+'layer_4'+suffix+'/kernel'),
                          bias_initializer=init_loader.init(prefix+'layer_4'+suffix+'/bias'),
                          use_bias=True, activation=None, trainable=conv_trainable)

    self.skip_level = skip_level

  def call(self, inputs):
    x0 = self.layer_1(inputs)
    x1 = self.layer_2(x0)
    x2 = self.layer_3(x1)
    x2 = tf.depth_to_space(x2, 2)
    x3 = self.layer_4(x2)
    return x3

class NeighborSample(tf.keras.layers.Layer):
  def __init__(self, in_shape, **kwargs):
    super(NeighborSample, self).__init__(**kwargs)
    self.in_shape = in_shape

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    channel_axis = -1
    dim = input_shape[channel_axis].value
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      flt = np.zeros((5, 5, dim, dim*25), dtype=np.float32)
      for i in range(0, 5):
        for j in range(0, 5):
          for k in range(0, dim):
            s = k*25 + i * 5 + j
            flt[i, j, k, s] = 1 
      self.sampler_filter = tf.constant(flt)

  def call(self, tensor):
    tensor = tf.nn.conv2d(tensor, self.sampler_filter, strides=[1,1,1,1], padding='SAME', name='bsample')
    b, h, w, c = self.in_shape
    tensor = tf.reshape(tensor, [b, h, w, c, 5, 5])
    tensor = tf.transpose(tensor, [0, 1, 2, 4, 5, 3])
    tensor = tf.reshape(tensor, [b*h*w, 5, 5, c])
    return tensor

class GetZSigma(tf.keras.layers.Layer):
  def __init__(self, dim, trainable=True, initializer=tf.keras.initializers.VarianceScaling(), **kwargs):
    super(GetZSigma, self).__init__(**kwargs)
    self.dim = dim
    self.initializer = initializer
    self.trainable = trainable

  def build(self, input_shape):
    with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
      self.z_sigma = self.add_variable('z_sigma',
        shape=(self.dim,),
        initializer=self.initializer,
        trainable=self.trainable
      )

  def call(self, tensor):
    return tf.abs(self.z_sigma)

class GaussianModel(tf.keras.layers.Layer):
  """Layer for a dependent Gaussian model, inherited from EntropyBottleneck"""
  def __init__(self, **kwargs):
    super(GaussianModel, self).__init__(**kwargs)
    # self.m_std_tensor = std_tensor
    self.m_normal_dist = tf.distributions.Normal(loc=0., scale=1.)
    self.likelihood_bound = 1e-9
    self.data_format = "channels_last"

  def _channel_axis(self, ndim):
    try:
      return {"channels_first": 1, "channels_last": ndim - 1}[self.data_format]
    except KeyError:
      raise ValueError("Unsupported `data_format` for {} layer: {}.".format(
          self.__class__.__name__, self.data_format))

  def build(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    channel_axis = self._channel_axis(input_shape.ndims)
    channels = input_shape[channel_axis].value
    if channels is None:
      raise ValueError("The channel dimension of the inputs must be defined.")
    self.input_spec = tf.keras.layers.InputSpec(
        ndim=input_shape.ndims, axes={channel_axis: channels})

    super(GaussianModel, self).build(input_shape)

  def _cumulative(self, inputs, stds, mu, stop_gradient=False):
    half = tf.constant(.5, dtype=self.dtype)
    eps = tf.constant(1e-6, dtype=tf.float32)
    upper = (inputs - mu + half) / (stds)
    lower = (inputs - mu - half) / (stds)
    cdf_upper = self.m_normal_dist.cdf(upper)
    cdf_lower = self.m_normal_dist.cdf(lower)
    res = cdf_upper - cdf_lower
    return res

  def call(self, inputs, hyper_sigma, hyper_mu, training):
    inputs = tf.convert_to_tensor(inputs)
    ndim = self.input_spec.ndim
    channel_axis = self._channel_axis(ndim)
    half = tf.constant(.5, dtype=self.dtype)
    values = inputs

    likelihood = self._cumulative(values, hyper_sigma, hyper_mu)

    if self.likelihood_bound > 0:
      likelihood_bound = tf.constant(
          self.likelihood_bound, dtype=self.dtype)
      likelihood = lower_bound(likelihood, likelihood_bound)

    return values, likelihood

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape)
    return input_shape, input_shape

class PredictionModelLoad(tf.keras.Model):
  def __init__(self, dim=192, trainable=True, outdim=None, suffix='', prefix='', init_loader=None):
    super(PredictionModelLoad, self).__init__()
    self.dim = dim
    P_trainable = trainable
    self.P_conv1 = Conv2D(dim, (3, 3), name='P_conv1%s' % (suffix),
      kernel_initializer=init_loader.init(prefix+'P_conv1%s/kernel' % (suffix)),
      bias_initializer=init_loader.init(prefix+'P_conv1%s/bias' % (suffix)),
      padding="SAME", use_bias=True, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.2), trainable=P_trainable)

    self.P_conv2 = Conv2D(dim, (3, 3), name='P_conv2%s' % (suffix),
      kernel_initializer=init_loader.init(prefix+'P_conv2%s/kernel' % (suffix)),
      bias_initializer=init_loader.init(prefix+'P_conv2%s/bias' % (suffix)),
      padding="SAME", strides=(2, 2), use_bias=True, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.2), trainable=P_trainable)

    self.P_conv3 = Conv2D(dim, (3, 3), name='P_conv3%s' % (suffix),
      kernel_initializer=init_loader.init(prefix+'P_conv3%s/kernel' % (suffix)),
      bias_initializer=init_loader.init(prefix+'P_conv3%s/bias' % (suffix)),
      padding="SAME", use_bias=True, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.2), trainable=P_trainable)

    if outdim is None:
      outdim = dim

    self.P_fc = tf.keras.layers.Dense(outdim*2, name='P_fc%s' % (suffix),
      kernel_initializer=init_loader.init(prefix+'P_fc%s/kernel' % (suffix)),
      bias_initializer=init_loader.init(prefix+'P_fc%s/bias' % (suffix)),
      use_bias=True, activation=None, trainable=P_trainable)
  
  def call(self, y_rounded, h_tilde, sampler, input_shape):

    b, h, w, c = input_shape
    h_sampled = sampler(h_tilde)
    h_sampled = self.P_conv1(h_sampled)
    h_sampled = self.P_conv2(h_sampled)
    h_sampled = self.P_conv3(h_sampled)
    h_sampled = tf.transpose(h_sampled, (0, 3, 1, 2))
    h_sampled = tf.reshape(h_sampled, (b*h*w, 9*self.dim))
    h_sampled = self.P_fc(h_sampled)

    hyper_mu = h_sampled[:, :c]
    hyper_mu = tf.reshape(hyper_mu, (b, h, w, c))
    hyper_sigma = h_sampled[:, c:]
    hyper_sigma = tf.exp(hyper_sigma)
    hyper_sigma = tf.reshape(hyper_sigma, (b, h, w, c))

    return hyper_mu, hyper_sigma

class SideInfoReconModelLoad(tf.keras.Model):
  def __init__(self, num_filters=192, conv_trainable=True, suffix='', init_loader=None):
    super(SideInfoReconModelLoad, self).__init__()
    self.layer_1 = Conv2DTranspose(num_filters, (5, 5), strides=(2, 2), padding="SAME", name='layer_1%s' % (suffix),
                          kernel_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_1%s/kernel' % (suffix)),
                          bias_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_1%s/bias' % (suffix)),
                          use_bias=True, activation=None, trainable=conv_trainable)
    self.layer_1a = Conv2DTranspose(num_filters, (5, 5), strides=(2,2), padding="SAME", name='layer_1a%s' % (suffix),
                          kernel_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_1a%s/kernel' % (suffix)),
                          bias_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_1a%s/bias' % (suffix)),
                          use_bias=True, activation=lambda x:tf.nn.leaky_relu(x, 0.2), trainable=conv_trainable)
    self.layer_1b = Conv2DTranspose(num_filters, (5, 5), strides=(2,2), padding="SAME", name='layer_1b%s' % (suffix),
                          kernel_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_1b%s/kernel' % (suffix)),
                          bias_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_1b%s/bias' % (suffix)),
                          use_bias=True, activation=lambda x:tf.nn.leaky_relu(x, 0.2), trainable=conv_trainable)

    self.layer_3_1 = Conv2D(num_filters, (3, 3), padding="SAME", name='layer_3_1%s' % (suffix),
                          kernel_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_3_1%s/kernel' % (suffix)),
                          bias_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_3_1%s/bias' % (suffix)),
                          use_bias=True, activation=lambda x:tf.nn.leaky_relu(x, 0.2), trainable=conv_trainable)
    self.layer_3_2 = Conv2D(num_filters, (3, 3), padding="SAME", name='layer_3_2%s' % (suffix),
                          kernel_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_3_2%s/kernel' % (suffix)),
                          bias_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_3_2%s/bias' % (suffix)),
                          use_bias=True, activation=lambda x:tf.nn.leaky_relu(x, 0.2), trainable=conv_trainable)
    self.layer_3_3 = Conv2D(num_filters*2, (3, 3), padding="SAME", name='layer_3_3%s' % (suffix),
                          kernel_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_3_3%s/kernel' % (suffix)),
                          bias_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_3_3%s/bias' % (suffix)),
                          use_bias=True, activation=lambda x:tf.nn.leaky_relu(x, 0.2), trainable=conv_trainable)


    self.layer_4 = Conv2DTranspose(num_filters//3, (5, 5), strides=(2,2), padding="SAME", name='layer_4%s' % (suffix),
                          kernel_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_4%s/kernel' % (suffix)),
                          bias_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_4%s/bias' % (suffix)),
                          use_bias=True, activation=None, trainable=conv_trainable)

    self.layer_5 = Conv2D(num_filters//12, (3, 3), padding="SAME", name='layer_5%s' % (suffix),
                          kernel_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_5%s/kernel' % (suffix)),
                          bias_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_5%s/bias' % (suffix)),
                          use_bias=True, activation=lambda x:tf.nn.leaky_relu(x, 0.2), trainable=conv_trainable)

    self.layer_6 = Conv2D(3, (1, 1), padding="SAME", name='layer_6%s' % (suffix),
                          kernel_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_6%s/kernel' % (suffix)),
                          bias_initializer=init_loader.init('RE6_GPU0/side_info_recon_model_load/layer_6%s/bias' % (suffix)),
                          use_bias=True, activation=None, trainable=conv_trainable)

  def call(self, pf, h2, h1):
    h1prime = tf.depth_to_space(h1, 2)
    h = tf.concat([h2, h1prime], -1)
    h = self.layer_1(h)
    h = self.layer_1a(h)
    h = self.layer_1b(h)

    hfeat_0 = tf.concat([pf, h], -1)
    hfeat = self.layer_3_1(hfeat_0)
    hfeat = self.layer_3_2(hfeat)
    hfeat = self.layer_3_3(hfeat)
    hfeat = tf.add_n([hfeat_0, hfeat])

    x = self.layer_4(hfeat)
    x = self.layer_5(x)
    x = self.layer_6(x)
    return x

def compress_low(args):
  """Compresses an image."""
  from PIL import Image
  # Load input image and add batch dimension.
  f = Image.open(args.input)
  fshape = [f.size[1], f.size[0], 3]
  x = load_image(args.input)
  x = tf.expand_dims(x, 0)

  compressed_file_path = args.output
  fileobj = open(compressed_file_path, mode='wb')

  qp = args.qp
  model_type = args.model_type
  print(f'model_type: {model_type}, qp: {qp}')
  l_init = LoadInitializer(f'./models/model{model_type}_qp{qp}.pk')

  buf = qp << 1
  buf = buf + model_type
  arr = np.array([0], dtype=np.uint8)
  arr[0] =  buf
  arr.tofile(fileobj)

  h, w = (fshape[0]//64)*64, (fshape[1]//64)*64
  if h < fshape[0]:
    h += 64
  if w < fshape[1]:
    w += 64

  pad_up = (h - fshape[0]) // 2
  pad_down = (h - fshape[0]) - pad_up
  pad_left = (w - fshape[1]) // 2
  pad_right = (w - fshape[1]) - pad_left

  x = tf.pad(x, [[0, 0], [pad_up, pad_down], [pad_left, pad_right], [0, 0]])

  x.set_shape([1, ]+list([h, w, 3]))
  fshape = [h, w, 3]
  in_x = x

  arr = np.array([w, h], dtype=np.uint16)
  arr.tofile(fileobj)
  fileobj.close()

  a_model = analysisTransformModel(192, conv_trainable=False, gdn_trainable=False, suffix='_mainA', init_loader=l_init)
  s_model = synthesisTransformModel(192, conv_trainable=False, gdn_trainable=False, suffix='_mainS', init_loader=l_init)

  ha_model_1 = h_analysisTransformModelLoad([64*4,32*4,32*4], [(1,1), (1,1), (1,1)],
                    conv_trainable=False, suffix='_h1a', prefix='RE6_GPU0/h_analysis_transform_model_load/', init_loader=l_init)
  hs_model_1 = h_synthesisTransformModelLoad([64*4,64*4,64*4], [(1,1), (1,1), (1,1)],
                    conv_trainable=False, suffix='_h1s', prefix='RE6_GPU0/h_synthesis_transform_model_load/', init_loader=l_init)

  ha_model_2 = h_analysisTransformModelLoad([384,192*4,64*4], [(1,1), (1,1), (1,1)],
                    conv_trainable=False, suffix='_h2a', prefix='RE6_GPU0/h_analysis_transform_model_load_1/', init_loader=l_init)
  hs_model_2 = h_synthesisTransformModelLoad([192*4,192*4,192], [(1,1), (1,1), (1,1)],
                    conv_trainable=False, suffix='_h2s', prefix='RE6_GPU0/h_synthesis_transform_model_load_1/', init_loader=l_init)


  entropy_bottleneck_1 = GaussianModel()
  entropy_bottleneck_2 = GaussianModel()
  entropy_bottleneck_3 = GaussianModel()

  get_h1_sigma = GetZSigma(32*4, initializer=l_init.init('RE6_GPU0/get_h1_sigma/get_h1_sigma/z_sigma'), trainable=False, name='get_h1_sigma')
  prediction_model_2 = PredictionModelLoad(64*4, outdim=64*4, suffix='_pred_2',
                  trainable=False, prefix='RE6_GPU0/prediction_model_load/', init_loader=l_init)
  prediction_model_3 = PredictionModelLoad(192, outdim=192, suffix='_pred_3',
                  trainable=False, prefix='RE6_GPU0/prediction_model_load_1/', init_loader=l_init)

  b, h, w, c = 1, fshape[0]//16, fshape[1]//16, 192

  sampler_2 = NeighborSample((b,h//2,w//2,64*4))
  sampler_3 = NeighborSample((b,h,w,c))

  side_recon_model = SideInfoReconModelLoad(suffix='_mainS_recon', init_loader=l_init)
  

  with tf.name_scope('RE6_GPU%d' % (0)) as scope:
    test_num_pixels = fshape[0] * fshape[1]

    y = a_model(in_x)

    z3 = y
    z3_rounded = tf.round(z3)

    z2 = ha_model_2(z3_rounded)
    z2_rounded = tf.round(z2)

    z1 = ha_model_1(z2_rounded)
    z1_rounded = tf.round(z1)

    z1_sigma = get_h1_sigma(None)
    z1_mu = tf.zeros_like(z1_sigma)
    
    with tf.device('/cpu:0'):
      _, test_z1_likelihoods = entropy_bottleneck_1(z1_rounded, z1_sigma, z1_mu, training=False)
    h1 = hs_model_1(z1_rounded) # 192

    cond_1 = h1
    z2_mu, z2_sigma = prediction_model_2(z2_rounded, cond_1, sampler_2, (b,h//2,w//2,64*4))
    
    with tf.device('/cpu:0'):
      _, test_z2_likelihoods = entropy_bottleneck_2(z2_rounded, z2_sigma, z2_mu, training=False)
    h2 = hs_model_2(z2_rounded)

    cond_2 = h2
    z3_mu, z3_sigma = prediction_model_3(z3_rounded, cond_2, sampler_3, (b,h,w,c))
    with tf.device('/cpu:0'):
      _, test_z3_likelihoods = entropy_bottleneck_3(z3_rounded, z3_sigma, z3_mu, training=False)

    
    x_tilde_org, pf = s_model(z3_rounded)
    x_tilde = side_recon_model(pf, h2, h1)
    
    eval_bpp = tf.reduce_sum(tf.log(test_z3_likelihoods)) / (-np.log(2) * test_num_pixels) + \
               tf.reduce_sum(tf.log(test_z2_likelihoods)) / (-np.log(2) * test_num_pixels) + \
               tf.reduce_sum(tf.log(test_z1_likelihoods)) / (-np.log(2) * test_num_pixels)
    
    with tf.Session() as sess:
      begin_time = time.time()
      sess.run(tf.global_variables_initializer())
      
      initial_time = time.time()
      v_z1_sigma, v_z1_mu, v_z2_sigma, v_z2_mu, v_z3_sigma, v_z3_mu, v_z1_rounded, v_z2_rounded, v_z3_rounded, v_bpp = sess.run(
        [z1_sigma, z1_mu, z2_sigma, z2_mu, z3_sigma, z3_mu, z1_rounded, z2_rounded, z3_rounded, eval_bpp]
        )
      exec_time = time.time()

      
      
      bitout = arithmeticcoding.BitOutputStream(open(compressed_file_path, "ab+"))
      enc = arithmeticcoding.ArithmeticEncoder(bitout)

      for ch_idx in range(v_z1_rounded.shape[-1]):
        mu_val = 255
        sigma_val = v_z1_sigma[ch_idx]

        freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

        for h_idx in range(v_z1_rounded.shape[1]):
          for w_idx in range(v_z1_rounded.shape[2]):
            symbol = np.int(v_z1_rounded[0, h_idx, w_idx, ch_idx] + 255)
            if symbol < 0 or symbol > 511:
              print("symbol range error: " + str(symbol))

            enc.write(freq, symbol)

      
      for h_idx in range(v_z2_rounded.shape[1]):
        for w_idx in range(v_z2_rounded.shape[2]):
          for ch_idx in range(v_z2_rounded.shape[-1]):
            symbol = np.int(v_z2_rounded[0, h_idx, w_idx, ch_idx] + 255)
            if symbol < 0 or symbol > 511:
              print("symbol range error: " + str(symbol))
            mu_val = v_z2_mu[0,h_idx,w_idx,ch_idx] + 255
            sigma_val = v_z2_sigma[0,h_idx,w_idx,ch_idx]
            freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

            enc.write(freq, symbol)

      for h_idx in range(v_z3_rounded.shape[1]):
        for w_idx in range(v_z3_rounded.shape[2]):
          for ch_idx in range(v_z3_rounded.shape[-1]):
            symbol = np.int(v_z3_rounded[0, h_idx, w_idx, ch_idx] + 255)
            if symbol < 0 or symbol > 511:
              print("symbol range error: " + str(symbol))
            mu_val = v_z3_mu[0,h_idx,w_idx,ch_idx] + 255
            sigma_val = v_z3_sigma[0,h_idx,w_idx,ch_idx]
            freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

            enc.write(freq, symbol)

      enc.write(freq, 512)
      enc.finish()
      bitout.close()
      write_time = time.time()

      print("Estimated bpp: %.5f" % (v_bpp))
      print("Execution time: #1 %.4f #2 %.4f #3 %.4f" % (initial_time - begin_time, exec_time - initial_time, write_time - exec_time))

      if args.save_recon:
        fid = args.output.split('/')[-1][:-4]
        if pad_down != 0 and pad_right != 0:
          sess.run(save_image('recon_%s.png' % (fid), x_tilde[0, pad_up:-pad_down,pad_left:-pad_right, :]))
        elif pad_down != 0:
          sess.run(save_image('recon_%s.png' % (fid), x_tilde[0, pad_up:-pad_down,pad_left:, :]))
        elif pad_right != 0:
          sess.run(save_image('recon_%s.png' % (fid), x_tilde[0, pad_up:,pad_left:-pad_right, :]))
        else:
          sess.run(save_image('recon_%s.png' % (fid), x_tilde[0, pad_up:,pad_left:, :]))

      return compressed_file_path

def decompress_low(args):
  from PIL import Image

  compressed_file = args.input
  fileobj = open(compressed_file, mode='rb')
  buf = fileobj.read(1)
  arr = np.frombuffer(buf, dtype=np.uint8)
  model_type = arr[0] % 2
  qp = arr[0] >> 1
  if qp > 3:
    fileobj.close()
    return
  print(f'model_type: {model_type}, qp: {qp}')
  l_init = LoadInitializer(f'./models/model{model_type}_qp{qp}.pk')

  buf = fileobj.read(4)
  arr = np.frombuffer(buf, dtype=np.uint16)
  w = int(arr[0])
  h = int(arr[1])
  oh = h
  ow = w

  fshape = [h, w]

  h, w = (fshape[0]//64)*64, (fshape[1]//64)*64
  if h < fshape[0]:
    h += 64
  if w < fshape[1]:
    w += 64

  pad_up = (h - fshape[0]) // 2
  pad_down = (h - fshape[0]) - pad_up
  pad_left = (w - fshape[1]) // 2
  pad_right = (w - fshape[1]) - pad_left


  c = 192
  b = 1
  padded_w = int(math.ceil(w / 16) * 16)
  padded_h = int(math.ceil(h / 16) * 16)

  # Read the shape information and compressed string from the binary file.
  with tf.name_scope('RE6_GPU%d' % (0)) as scope:
    
    s_model = synthesisTransformModel(192, conv_trainable=False, gdn_trainable=False, suffix='_mainS', init_loader=l_init)

    
    hs_model_1 = h_synthesisTransformModelLoad([64*4,64*4,64*4], [(1,1), (1,1), (1,1)],
                      conv_trainable=False, suffix='_h1s', prefix='RE6_GPU0/h_synthesis_transform_model_load/', init_loader=l_init)

    
    hs_model_2 = h_synthesisTransformModelLoad([192*4,192*4,192], [(1,1), (1,1), (1,1)],
                      conv_trainable=False, suffix='_h2s', prefix='RE6_GPU0/h_synthesis_transform_model_load_1/', init_loader=l_init)


    entropy_bottleneck_1 = GaussianModel()
    entropy_bottleneck_2 = GaussianModel()
    entropy_bottleneck_3 = GaussianModel()

    get_h1_sigma = GetZSigma(32*4, initializer=l_init.init('RE6_GPU0/get_h1_sigma/get_h1_sigma/z_sigma'), trainable=False, name='get_h1_sigma')

    prediction_model_2 = PredictionModelLoad(64*4, outdim=64*4, suffix='_pred_2',
                    trainable=False, prefix='RE6_GPU0/prediction_model_load/', init_loader=l_init) # corresponds to code width of z2

    prediction_model_3 = PredictionModelLoad(192, outdim=192, suffix='_pred_3',
                    trainable=False, prefix='RE6_GPU0/prediction_model_load_1/', init_loader=l_init) # corresponds to code width of z3

    b, h, w, c = 1, padded_h//16, padded_w//16, 192

    sampler_2 = NeighborSample((b,h//2,w//2,64*4))
    sampler_3 = NeighborSample((b,h,w,c))

    side_recon_model = SideInfoReconModelLoad(suffix='_mainS_recon', init_loader=l_init)

    z1_rounded = tf.placeholder(dtype=tf.float32, shape=(1, padded_h//64, padded_w//64, 32*4))

    z2_rounded = tf.placeholder(dtype=tf.float32, shape=(1, padded_h//32, padded_w//32, 64*4))

    z3_rounded = tf.placeholder(dtype=tf.float32, shape=(1, padded_h//16, padded_w//16, 192))

    z1_sigma = get_h1_sigma(None)
    z1_mu = tf.zeros_like(z1_sigma)
    pred_0_1 = z1_mu
    
    
    h1 = hs_model_1(z1_rounded) # 192

    cond_1 = h1
    z2_mu, z2_sigma = prediction_model_2(z2_rounded, cond_1, sampler_2, (b, h//2, w//2, 64*4)) # corresponds to the to-predict width
    pred_1_2 = z2_mu
    
    h2 = hs_model_2(z2_rounded)


    cond_2 = h2
    z3_mu, z3_sigma = prediction_model_3(z3_rounded, cond_2, sampler_3, (b, h, w, c)) # corresponds to the to-predict width
    pred_2_3 = z3_mu

    x_tilde_org, pf = s_model(z3_rounded)
    x_tilde = side_recon_model(pf, h2, h1)

    ###################################################################
    
    with tf.Session() as sess:
      # tf.train.Saver().restore(sess, save_path=args.test_ckpt)
      begin_time = time.time()
      sess.run(tf.global_variables_initializer())
      initial_time = time.time()

      sigma_z = sess.run(z1_sigma)

      ############### decode zhat ####################################
      bitin = arithmeticcoding.BitInputStream(fileobj)
      dec = arithmeticcoding.ArithmeticDecoder(bitin)

      z1_hat = np.zeros((1, padded_h//64, padded_w//64, 32*4), dtype=np.float32)
      z2_hat = np.zeros((1, padded_h//32, padded_w//32, 64*4), dtype=np.float32)
      z3_hat = np.zeros((1, padded_h//16, padded_w//16, 192), dtype=np.float32)

      for ch_idx in range(z1_hat.shape[-1]):
        mu_val = 255
        sigma_val = sigma_z[ch_idx]
        freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

        for h_idx in range(z1_hat.shape[1]):
          for w_idx in range(z1_hat.shape[2]):
            symbol = dec.read(freq)
            if symbol == 512:  # EOF symbol
              print("EOF symbol")
              break
            z1_hat[:, h_idx, w_idx, ch_idx] = symbol - 255
      
      decode_z1_time = time.time()
      
      v_z2_mu, v_z2_sigma = sess.run([z2_mu, z2_sigma], feed_dict={z1_rounded:z1_hat})

      z2_exec_time = time.time()
      
      for h_idx in range(z2_hat.shape[1]):
        for w_idx in range(z2_hat.shape[2]):
          for ch_idx in range(z2_hat.shape[-1]):
            mu_val = v_z2_mu[0,h_idx,w_idx,ch_idx] + 255
            sigma_val = v_z2_sigma[0,h_idx,w_idx,ch_idx]
            freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)
            symbol = dec.read(freq)
            if symbol == 512:  # EOF symbol
              print("EOF symbol")
              break
            z2_hat[:, h_idx, w_idx, ch_idx] = symbol - 255

      decode_z2_time = time.time()
      v_z3_mu, v_z3_sigma = sess.run([z3_mu, z3_sigma], feed_dict={z2_rounded:z2_hat})
      z3_exec_time = time.time()
      for h_idx in range(z3_hat.shape[1]):
        for w_idx in range(z3_hat.shape[2]):
          for ch_idx in range(z3_hat.shape[-1]):
            mu_val = v_z3_mu[0,h_idx,w_idx,ch_idx] + 255
            sigma_val = v_z3_sigma[0,h_idx,w_idx,ch_idx]
            freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)
            symbol = dec.read(freq)
            if symbol == 512:  # EOF symbol
              print("EOF symbol")
              break
            z3_hat[:, h_idx, w_idx, ch_idx] = symbol - 255            
      decode_z3_time = time.time()
      if pad_down != 0 and pad_right != 0:
        recon = sess.run(save_image(args.output, x_tilde[0, pad_up:-pad_down,pad_left:-pad_right, :]), feed_dict={z1_rounded: z1_hat, z2_rounded:z2_hat, z3_rounded:z3_hat})
      elif pad_down != 0:
        recon = sess.run(save_image(args.output, x_tilde[0, pad_up:-pad_down,pad_left:, :]), feed_dict={z1_rounded: z1_hat, z2_rounded:z2_hat, z3_rounded:z3_hat})
      elif pad_right != 0:
        recon = sess.run(save_image(args.output, x_tilde[0, pad_up:,pad_left:-pad_right, :]), feed_dict={z1_rounded: z1_hat, z2_rounded:z2_hat, z3_rounded:z3_hat})
      else:
        recon = sess.run(save_image(args.output, x_tilde[0, pad_up:,pad_left:, :]), feed_dict={z1_rounded: z1_hat, z2_rounded:z2_hat, z3_rounded:z3_hat})

      z3_final_time = time.time()
      print("Execution time: #init %.4f #decode %.4f #esti %.4f #decode %.4f #esti %.4f #decode %.4f #recon %.4f" % (initial_time - begin_time, decode_z1_time - initial_time, z2_exec_time - decode_z1_time, decode_z2_time - z2_exec_time, z3_exec_time - decode_z2_time, decode_z3_time - z3_exec_time, z3_final_time - decode_z3_time))

def compress_high(args):
  """Compresses an image."""
  from PIL import Image
  # Load input image and add batch dimension.
  f = Image.open(args.input)
  fshape = [f.size[1], f.size[0], 3]
  x = load_image(args.input)
  x = tf.expand_dims(x, 0)

  compressed_file_path = args.output
  fileobj = open(compressed_file_path, mode='wb')

  qp = args.qp
  model_type = args.model_type
  
  l_init = LoadInitializer(f'./models/model{model_type}_qp{qp}.pk')

  buf = qp << 1
  buf = buf + model_type
  print(f'model_type: {model_type}, qp: {qp}')
  arr = np.array([0], dtype=np.uint8)
  arr[0] =  buf
  arr.tofile(fileobj)

  h, w = (fshape[0]//64)*64, (fshape[1]//64)*64
  if h < fshape[0]:
    h += 64
  if w < fshape[1]:
    w += 64

  pad_up = (h - fshape[0]) // 2
  pad_down = (h - fshape[0]) - pad_up
  pad_left = (w - fshape[1]) // 2
  pad_right = (w - fshape[1]) - pad_left

  x = tf.pad(x, [[0, 0], [pad_up, pad_down], [pad_left, pad_right], [0, 0]])

  x.set_shape([1, ]+list([h, w, 3]))
  fshape = [h, w, 3]
  in_x = x

  arr = np.array([w, h], dtype=np.uint16)
  arr.tofile(fileobj)
  fileobj.close()

  a_model = analysisTransformModel(384, conv_trainable=False, gdn_trainable=False, suffix='_mainA', init_loader=l_init)
  s_model = synthesisTransformModel(384, conv_trainable=False, gdn_trainable=False, suffix='_mainS', init_loader=l_init)

  ha_model_1 = h_analysisTransformModelLoad([64*4*2,32*4*2,32*4], [(1,1), (1,1), (1,1)], suffix='_h1a', conv_trainable=False, prefix='RE6_GPU0/h_analysis_transform_model_load/', init_loader=l_init)
  hs_model_1 = h_synthesisTransformModelLoad([64*4*2,64*4*2,64*4], [(1,1), (1,1), (1,1)], suffix='_h1s', conv_trainable=False, prefix='RE6_GPU0/h_synthesis_transform_model_load/', init_loader=l_init)

  ha_model_2 = h_analysisTransformModelLoad([384*2,192*4*2,64*4], [(1,1), (1,1), (1,1)], suffix='_h2a', conv_trainable=False, prefix='RE6_GPU0/h_analysis_transform_model_load_1/', init_loader=l_init)
  hs_model_2 = h_synthesisTransformModelLoad([192*4*2,192*4*2,384], [(1,1), (1,1), (1,1)], suffix='_h2s', conv_trainable=False, prefix='RE6_GPU0/h_synthesis_transform_model_load_1/', init_loader=l_init)


  entropy_bottleneck_1 = GaussianModel()
  entropy_bottleneck_2 = GaussianModel()
  entropy_bottleneck_3 = GaussianModel()

  get_h1_sigma = GetZSigma(32*4, initializer=l_init.init('RE6_GPU0/get_h1_sigma/get_h1_sigma/z_sigma'), trainable=False, name='get_h1_sigma')
  prediction_model_2 = PredictionModelLoad(64*4, outdim=64*4, suffix='_pred_2',
                  trainable=False, prefix='RE6_GPU0/prediction_model_load/', init_loader=l_init) # corresponds to code width of z2
  prediction_model_3 = PredictionModelLoad(384, outdim=384, suffix='_pred_3',
                  trainable=False, prefix='RE6_GPU0/prediction_model_load_1/', init_loader=l_init) # corresponds to code width of z3

  b, h, w, c = 1, fshape[0]//16, fshape[1]//16, 384

  sampler_2 = NeighborSample((b,h//2,w//2,64*4))
  sampler_3 = NeighborSample((b,h,w,c))

  side_recon_model = SideInfoReconModelLoad(suffix='_mainS_recon', init_loader=l_init, num_filters=384)
  

  with tf.name_scope('RE6_GPU%d' % (0)) as scope:
    test_num_pixels = fshape[0] * fshape[1]

    y = a_model(in_x)

    z3 = y
    z3_rounded = tf.round(z3)

    z2 = ha_model_2(z3_rounded)
    z2_rounded = tf.round(z2)

    z1 = ha_model_1(z2_rounded)
    z1_rounded = tf.round(z1)

    z1_sigma = get_h1_sigma(None)
    z1_mu = tf.zeros_like(z1_sigma)
    
    with tf.device('/cpu:0'):
      _, test_z1_likelihoods = entropy_bottleneck_1(z1_rounded, z1_sigma, z1_mu, training=False)
    h1 = hs_model_1(z1_rounded) # 192

    cond_1 = h1
    z2_mu, z2_sigma = prediction_model_2(z2_rounded, cond_1, sampler_2, (b,h//2,w//2,64*4))
    
    with tf.device('/cpu:0'):
      _, test_z2_likelihoods = entropy_bottleneck_2(z2_rounded, z2_sigma, z2_mu, training=False)
    h2 = hs_model_2(z2_rounded)

    cond_2 = h2
    z3_mu, z3_sigma = prediction_model_3(z3_rounded, cond_2, sampler_3, (b,h,w,c))
    with tf.device('/cpu:0'):
      _, test_z3_likelihoods = entropy_bottleneck_3(z3_rounded, z3_sigma, z3_mu, training=False)

    
    x_tilde_org, pf = s_model(z3_rounded)
    x_tilde = side_recon_model(pf, h2, h1)
    
    eval_bpp = tf.reduce_sum(tf.log(test_z3_likelihoods)) / (-np.log(2) * test_num_pixels) + \
               tf.reduce_sum(tf.log(test_z2_likelihoods)) / (-np.log(2) * test_num_pixels) + \
               tf.reduce_sum(tf.log(test_z1_likelihoods)) / (-np.log(2) * test_num_pixels)
    
    with tf.Session() as sess:
      begin_time = time.time()
      sess.run(tf.global_variables_initializer())
      
      initial_time = time.time()
      v_z1_sigma, v_z1_mu, v_z2_sigma, v_z2_mu, v_z3_sigma, v_z3_mu, v_z1_rounded, v_z2_rounded, v_z3_rounded, v_bpp = sess.run(
        [z1_sigma, z1_mu, z2_sigma, z2_mu, z3_sigma, z3_mu, z1_rounded, z2_rounded, z3_rounded, eval_bpp]
        )
      exec_time = time.time()

      
      
      bitout = arithmeticcoding.BitOutputStream(open(compressed_file_path, "ab+"))
      enc = arithmeticcoding.ArithmeticEncoder(bitout)

      for ch_idx in range(v_z1_rounded.shape[-1]):
        mu_val = 255
        sigma_val = v_z1_sigma[ch_idx]

        freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

        for h_idx in range(v_z1_rounded.shape[1]):
          for w_idx in range(v_z1_rounded.shape[2]):
            symbol = np.int(v_z1_rounded[0, h_idx, w_idx, ch_idx] + 255)
            if symbol < 0 or symbol > 511:
              print("symbol range error: " + str(symbol))

            enc.write(freq, symbol)

      
      for h_idx in range(v_z2_rounded.shape[1]):
        for w_idx in range(v_z2_rounded.shape[2]):
          for ch_idx in range(v_z2_rounded.shape[-1]):
            symbol = np.int(v_z2_rounded[0, h_idx, w_idx, ch_idx] + 255)
            if symbol < 0 or symbol > 511:
              print("symbol range error: " + str(symbol))
            mu_val = v_z2_mu[0,h_idx,w_idx,ch_idx] + 255
            sigma_val = v_z2_sigma[0,h_idx,w_idx,ch_idx]
            freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

            enc.write(freq, symbol)

      for h_idx in range(v_z3_rounded.shape[1]):
        for w_idx in range(v_z3_rounded.shape[2]):
          for ch_idx in range(v_z3_rounded.shape[-1]):
            symbol = np.int(v_z3_rounded[0, h_idx, w_idx, ch_idx] + 255)
            if symbol < 0 or symbol > 511:
              print("symbol range error: " + str(symbol))
            mu_val = v_z3_mu[0,h_idx,w_idx,ch_idx] + 255
            sigma_val = v_z3_sigma[0,h_idx,w_idx,ch_idx]
            freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

            enc.write(freq, symbol)

      enc.write(freq, 512)
      enc.finish()
      bitout.close()
      write_time = time.time()

      print("Estimated bpp: %.5f" % (v_bpp))
      print("Execution time: #1 %.4f #2 %.4f #3 %.4f" % (initial_time - begin_time, exec_time - initial_time, write_time - exec_time))

      if args.save_recon:
        fid = args.output.split('/')[-1][:-4]
        if pad_down != 0 and pad_right != 0:
          sess.run(save_image('recon_%s.png' % (fid), x_tilde[0, pad_up:-pad_down,pad_left:-pad_right, :]))
        elif pad_down != 0:
          sess.run(save_image('recon_%s.png' % (fid), x_tilde[0, pad_up:-pad_down,pad_left:, :]))
        elif pad_right != 0:
          sess.run(save_image('recon_%s.png' % (fid), x_tilde[0, pad_up:,pad_left:-pad_right, :]))
        else:
          sess.run(save_image('recon_%s.png' % (fid), x_tilde[0, pad_up:,pad_left:, :]))

      return compressed_file_path

def decompress_high(args):
  from PIL import Image

  compressed_file = args.input
  fileobj = open(compressed_file, mode='rb')
  buf = fileobj.read(1)
  arr = np.frombuffer(buf, dtype=np.uint8)
  model_type = arr[0] % 2
  qp = arr[0] >> 1
  print(f'model_type: {model_type}, qp: {qp}')
  if qp <= 3:
    fileobj.close()
    return
  
  l_init = LoadInitializer(f'./models/model{model_type}_qp{qp}.pk')

  buf = fileobj.read(4)
  arr = np.frombuffer(buf, dtype=np.uint16)
  w = int(arr[0])
  h = int(arr[1])
  oh = h
  ow = w

  fshape = [h, w]

  h, w = (fshape[0]//64)*64, (fshape[1]//64)*64
  if h < fshape[0]:
    h += 64
  if w < fshape[1]:
    w += 64

  pad_up = (h - fshape[0]) // 2
  pad_down = (h - fshape[0]) - pad_up
  pad_left = (w - fshape[1]) // 2
  pad_right = (w - fshape[1]) - pad_left


  c = 384
  b = 1
  padded_w = int(math.ceil(w / 16) * 16)
  padded_h = int(math.ceil(h / 16) * 16)

  # Read the shape information and compressed string from the binary file.
  with tf.name_scope('RE6_GPU%d' % (0)) as scope:
    
    s_model = synthesisTransformModel(384, conv_trainable=False, gdn_trainable=False, suffix='_mainS', init_loader=l_init)

    
    hs_model_1 = h_synthesisTransformModelLoad([64*4*2,64*4*2,64*4], [(1,1), (1,1), (1,1)], suffix='_h1s', conv_trainable=False, prefix='RE6_GPU0/h_synthesis_transform_model_load/', init_loader=l_init)

    hs_model_2 = h_synthesisTransformModelLoad([192*4*2,192*4*2,384], [(1,1), (1,1), (1,1)], suffix='_h2s', conv_trainable=False, prefix='RE6_GPU0/h_synthesis_transform_model_load_1/', init_loader=l_init)


    entropy_bottleneck_1 = GaussianModel()
    entropy_bottleneck_2 = GaussianModel()
    entropy_bottleneck_3 = GaussianModel()

    get_h1_sigma = GetZSigma(32*4, initializer=l_init.init('RE6_GPU0/get_h1_sigma/get_h1_sigma/z_sigma'), trainable=False, name='get_h1_sigma')

    prediction_model_2 = PredictionModelLoad(64*4, outdim=64*4, suffix='_pred_2',
                    trainable=False, prefix='RE6_GPU0/prediction_model_load/', init_loader=l_init) # corresponds to code width of z2

    prediction_model_3 = PredictionModelLoad(384, outdim=384, suffix='_pred_3',
                    trainable=False, prefix='RE6_GPU0/prediction_model_load_1/', init_loader=l_init) # corresponds to code width of z3

    b, h, w, c = 1, padded_h//16, padded_w//16, 384

    sampler_2 = NeighborSample((b,h//2,w//2,64*4))
    sampler_3 = NeighborSample((b,h,w,c))

    side_recon_model = SideInfoReconModelLoad(suffix='_mainS_recon', init_loader=l_init, num_filters=384)

    z1_rounded = tf.placeholder(dtype=tf.float32, shape=(1, padded_h//64, padded_w//64, 32*4))

    z2_rounded = tf.placeholder(dtype=tf.float32, shape=(1, padded_h//32, padded_w//32, 64*4))

    z3_rounded = tf.placeholder(dtype=tf.float32, shape=(1, padded_h//16, padded_w//16, 384))

    z1_sigma = get_h1_sigma(None)
    z1_mu = tf.zeros_like(z1_sigma)
    pred_0_1 = z1_mu
    
    
    h1 = hs_model_1(z1_rounded) # 192

    cond_1 = h1
    z2_mu, z2_sigma = prediction_model_2(z2_rounded, cond_1, sampler_2, (b, h//2, w//2, 64*4)) # corresponds to the to-predict width
    pred_1_2 = z2_mu
    
    h2 = hs_model_2(z2_rounded)


    cond_2 = h2
    z3_mu, z3_sigma = prediction_model_3(z3_rounded, cond_2, sampler_3, (b, h, w, c)) # corresponds to the to-predict width
    pred_2_3 = z3_mu

    x_tilde_org, pf = s_model(z3_rounded)
    x_tilde = side_recon_model(pf, h2, h1)
    
    with tf.Session() as sess:
      # tf.train.Saver().restore(sess, save_path=args.test_ckpt)
      begin_time = time.time()
      sess.run(tf.global_variables_initializer())
      initial_time = time.time()

      sigma_z = sess.run(z1_sigma)

      ############### decode zhat ####################################
      bitin = arithmeticcoding.BitInputStream(fileobj)
      dec = arithmeticcoding.ArithmeticDecoder(bitin)

      z1_hat = np.zeros((1, padded_h//64, padded_w//64, 32*4), dtype=np.float32)
      z2_hat = np.zeros((1, padded_h//32, padded_w//32, 64*4), dtype=np.float32)
      z3_hat = np.zeros((1, padded_h//16, padded_w//16, 384), dtype=np.float32)

      for ch_idx in range(z1_hat.shape[-1]):
        mu_val = 255
        sigma_val = sigma_z[ch_idx]
        freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)

        for h_idx in range(z1_hat.shape[1]):
          for w_idx in range(z1_hat.shape[2]):
            symbol = dec.read(freq)
            if symbol == 512:  # EOF symbol
              print("EOF symbol")
              break
            z1_hat[:, h_idx, w_idx, ch_idx] = symbol - 255
      
      decode_z1_time = time.time()
      
      v_z2_mu, v_z2_sigma = sess.run([z2_mu, z2_sigma], feed_dict={z1_rounded:z1_hat})

      z2_exec_time = time.time()
      
      for h_idx in range(z2_hat.shape[1]):
        for w_idx in range(z2_hat.shape[2]):
          for ch_idx in range(z2_hat.shape[-1]):
            mu_val = v_z2_mu[0,h_idx,w_idx,ch_idx] + 255
            sigma_val = v_z2_sigma[0,h_idx,w_idx,ch_idx]
            freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)
            symbol = dec.read(freq)
            if symbol == 512:  # EOF symbol
              print("EOF symbol")
              break
            z2_hat[:, h_idx, w_idx, ch_idx] = symbol - 255

      decode_z2_time = time.time()
      v_z3_mu, v_z3_sigma = sess.run([z3_mu, z3_sigma], feed_dict={z2_rounded:z2_hat})
      z3_exec_time = time.time()
      for h_idx in range(z3_hat.shape[1]):
        for w_idx in range(z3_hat.shape[2]):
          for ch_idx in range(z3_hat.shape[-1]):
            mu_val = v_z3_mu[0,h_idx,w_idx,ch_idx] + 255
            sigma_val = v_z3_sigma[0,h_idx,w_idx,ch_idx]
            freq = arithmeticcoding.ModelFrequencyTable(mu_val, sigma_val)
            symbol = dec.read(freq)
            if symbol == 512:  # EOF symbol
              print("EOF symbol")
              break
            z3_hat[:, h_idx, w_idx, ch_idx] = symbol - 255            
      decode_z3_time = time.time()
      if pad_down != 0 and pad_right != 0:
        recon = sess.run(save_image(args.output, x_tilde[0, pad_up:-pad_down,pad_left:-pad_right, :]), feed_dict={z1_rounded: z1_hat, z2_rounded:z2_hat, z3_rounded:z3_hat})
      elif pad_down != 0:
        recon = sess.run(save_image(args.output, x_tilde[0, pad_up:-pad_down,pad_left:, :]), feed_dict={z1_rounded: z1_hat, z2_rounded:z2_hat, z3_rounded:z3_hat})
      elif pad_right != 0:
        recon = sess.run(save_image(args.output, x_tilde[0, pad_up:,pad_left:-pad_right, :]), feed_dict={z1_rounded: z1_hat, z2_rounded:z2_hat, z3_rounded:z3_hat})
      else:
        recon = sess.run(save_image(args.output, x_tilde[0, pad_up:,pad_left:, :]), feed_dict={z1_rounded: z1_hat, z2_rounded:z2_hat, z3_rounded:z3_hat})

      z3_final_time = time.time()
      print("Execution time: #init %.4f #decode %.4f #esti %.4f #decode %.4f #esti %.4f #decode %.4f #recon %.4f" % (initial_time - begin_time, decode_z1_time - initial_time, z2_exec_time - decode_z1_time, decode_z2_time - z2_exec_time, z3_exec_time - decode_z2_time, decode_z3_time - z3_exec_time, z3_final_time - decode_z3_time))
