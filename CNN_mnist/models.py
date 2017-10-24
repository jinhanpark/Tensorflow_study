import tensorflow as tf


class Model():
    def __init__(self, config, data):
        with tf.name_scope('Network') as scope:
            self.x = tf.placeholder(tf.float32, shape=[None, config.fan_in])
            self.y_ = tf.placeholder(tf.float32, shape=[None, config.fan_out])
            
            x_image = tf.reshape(self.x, [-1,28,28,1])
            
            with tf.name_scope('conv1') as scope:
                conv1 = tf.layers.conv2d(
                    inputs=x_image,
                    filters=32,
                    kernel_size=[5, 5],
                    padding="same",
                    activation=tf.nn.relu)
                
            with tf.name_scope('pool1') as scope:
                pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
                
            with tf.name_scope('conv2') as scope:
                conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=64,
                    kernel_size=[5, 5],
                    padding="same",
                    activation=tf.nn.relu)
                
            with tf.name_scope('pool2') as scope:
                pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
                pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
                
            with tf.name_scope('FC1') as scope:
                dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
                
            with tf.name_scope('dropout') as scope:
                self.keep_prob = tf.placeholder(tf.float32)
                dropout = tf.layers.dropout(inputs=dense, rate=self.keep_prob)
                
            with tf.name_scope('logits') as scope:
                self.logits = tf.layers.dense(inputs=dropout, units=config.fan_out)
                
            with tf.name_scope('loss') as scope:
                self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.logits))
