import tensorflow as tf
import numpy as np
import sys
from layer import *
        
class becalf:
    def __init__(self, x, downscaled, is_training, batch_size):
        self.batch_size = batch_size
        self.imitation = self.generator(downscaled, is_training, False)

    def generator(self, x_raw, is_training, reuse):
        with tf.variable_scope('generator', reuse=reuse):

            with tf.variable_scope('conv1'):
                x = conv_layer(x_raw, [3, 3, 3, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv2'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                x_raw = tf.concat([x,x_raw],3)
            with tf.variable_scope('conv3'):
                x = conv_layer(x_raw, [3, 3, 131, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv4'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                x_raw = tf.concat([x,x_raw],3)
            with tf.variable_scope('conv5'):
                x = conv_layer(x_raw, [3, 3, 259, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv6'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                x_raw = tf.concat([x,x_raw],3)

            with tf.variable_scope('conv7'):
                x = conv_layer(x_raw, [3, 3, 387, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('conv8'):
                x = conv_layer(x, [3, 3, 128, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                x_raw = tf.concat([x,x_raw],3)
                
            with tf.variable_scope('deconv9'):
                x = deconv_layer(x_raw, [3, 3, 128, 515], [self.batch_size, 436, 1024, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv10'):
                x = deconv_layer(x, [3, 3, 128, 128], [self.batch_size, 436, 1024, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                x_raw = tf.concat([x,x_raw],3)
            with tf.variable_scope('deconv11'):
                x = deconv_layer(x_raw, [3, 3, 128, 643], [self.batch_size, 436, 1024, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv12'):
                x = deconv_layer(x, [3, 3, 128, 128], [self.batch_size, 436, 1024, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                x_raw = tf.concat([x,x_raw],3)
            with tf.variable_scope('deconv13'):
                x = deconv_layer(x_raw, [3, 3, 128, 771], [self.batch_size, 436, 1024, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv14'):
                x = deconv_layer(x, [3, 3, 128, 128], [self.batch_size, 436, 1024, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
                x_raw = tf.concat([x,x_raw],3)
            with tf.variable_scope('deconv15'):
                x = deconv_layer(x_raw, [3, 3, 128, 899], [self.batch_size, 436, 1024, 128], 1)
                x = batch_normalize(x, is_training)
                x = tf.nn.relu(x)
            with tf.variable_scope('deconv16'):
                x = deconv_layer(x, [3, 3, 3, 128], [self.batch_size, 436, 1024, 3], 1)
        self.g_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
        return x



