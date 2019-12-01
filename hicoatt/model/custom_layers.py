'''
Created on 10.04.2019

@author: Philipp
'''
import tensorflow as tf
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.keras.layers import Layer


class BinaryMaskedSoftmax(_Merge):
    
    def _merge_function(self, inputs):
        # inputs order is important
        x = inputs[0]
        binary_mask = inputs[1]
        # ignore masked values for each sample
        x = tf.nn.softmax(x) * binary_mask
        # rescale_probability_mass for each sample
        x = x / tf.reduce_sum(x, axis=1, keepdims=True)
        return x


class BinaryMasking(Layer):
    
    def __init__(self, **kwargs):
        super(BinaryMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BinaryMasking, self).build(input_shape)

    def call(self, x):
        return tf.cast(tf.not_equal(x, 0), dtype="float32")


CUSTOM_LAYER_REGISTRY = {
            "BinaryMaskedSoftmax": BinaryMaskedSoftmax,
            "BinaryMasking": BinaryMasking
}
