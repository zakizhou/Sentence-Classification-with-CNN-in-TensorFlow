from __future__ import division
from __future__ import absolute_import
from __future__ import print_function


import tensorflow as tf


class Config(object):
    def __init__(self):
        self.embedding_size = 128
        self.kernel_sizes = [3, 4, 5]
        self.num_kernels = 128


class TextCNN(object):
    def __init__(self, config, inputs):
        embedding_size = config.embedding_size
        kernel_sizes = config.kernel_sizes
        num_kernels = config.num_kernels

        vocab_size = inputs.vocab_size
        sequence_length = inputs.sequence_length
        num_classes = inputs.num_classes
        with tf.variable_scope("embedding"):
            embedding = tf.get_variable("embedding",
                                        shape=[vocab_size, embedding_size],
                                        initializer=tf.truncated_normal_initializer(stddev=0.05),
                                        dtype=tf.float32)
            embed = tf.nn.embedding_lookup(embedding, inputs.inputs)
            expand = tf.expand_dims(embed, 3)

        outputs = []

        for i, kernel_size in enumerate(kernel_sizes):
            with tf.variable_scope("conv_pool_%d" % i):
                kernel = tf.get_variable("kernel",
                                         shape=[kernel_size, embedding_size, 1, num_kernels],
                                         initializer=tf.truncated_normal_initializer(stddev=0.05),
                                         dtype=tf.float32)
                bias = tf.get_variable("bias",
                                       shape=[num_kernels],
                                       initializer=tf.constant_initializer(value=0.),
                                       dtype=tf.float32)
                conv = tf.nn.conv2d(input=expand,
                                    filter=kernel,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID")
                conv_bias = tf.nn.bias_add(conv, bias)
                relu = tf.nn.relu(conv_bias)
                pool = tf.nn.max_pool(relu,
                                      ksize=[1, sequence_length - kernel_size + 1, 1, 1],
                                      strides=[1, 1, 1, 1],
                                      padding="VALID")
                outputs.append(pool)

        concat = tf.concat(3, outputs)
        squeeze = tf.squeeze(concat, squeeze_dims=[1, 2])
        dim = squeeze.get_shape().as_list()[-1]

        with tf.variable_scope("output"):
            softmax_w = tf.get_variable("softmax_w",
                                        shape=[dim, num_classes],
                                        initializer=tf.truncated_normal_initializer(stddev=0.05),
                                        dtype=tf.float32)
            softmax_b = tf.get_variable("softmax_b",
                                        shape=[num_classes],
                                        initializer=tf.constant_initializer(value=0.),
                                        dtype=tf.float32)
            logits = tf.nn.xw_plus_b(squeeze, softmax_w, softmax_b)
        with tf.name_scope("loss"):
            cross_entropy_per_example = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, inputs.labels)
            self.loss = tf.reduce_mean(cross_entropy_per_example)

        with tf.name_scope("train"):
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
            self.train = optimizer.minimize(self.loss)

        with tf.name_scope("validatin"):
            predict = tf.argmax(logits, 1)
            equal = tf.equal(predict, inputs.labels)
            self.validation = tf.reduce_mean(tf.cast(equal, tf.float32))