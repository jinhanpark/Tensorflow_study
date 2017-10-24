import numpy as np
import tensorflow as tf

class Model:
    def __init__(self, config, data):

        self.x = tf.placeholder(tf.float32)
        self.y_true = tf.placeholder(tf.float32)

        self.make_hidden_1(config)
        self.make_hidden_2(config)
        self.make_output(config)

        self.get_cost(config)

    def make_hidden_1(self, config):
        self.W1 = tf.Variable(tf.truncated_normal((config.input_size, config.hidden_size_1)), dtype=tf.float32, name="W1")
        self.b1 = tf.Variable(tf.zeros((1, config.hidden_size_1)), dtype=tf.float32, name="b1")
        self.h1 = tf.nn.relu(tf.add(tf.matmul(self.x, self.W1), self.b1))

    def make_hidden_2(self, config):
        self.W2 = tf.Variable(tf.truncated_normal((config.hidden_size_1, config.hidden_size_2)), dtype=tf.float32, name="W2")
        self.b2 = tf.Variable(tf.zeros((1, config.hidden_size_2)), dtype=tf.float32, name="b2")
        self.h2 = tf.nn.relu(tf.add(tf.matmul(self.h1, self.W2), self.b2))

    def make_output(self, config):
        self.W3 = tf.Variable(tf.truncated_normal((config.hidden_size_2, config.output_size)), dtype=tf.float32, name="W3")
        self.b3 = tf.Variable(tf.zeros((1, config.output_size)), dtype=tf.float32, name="b3")
        self.y = tf.add(tf.matmul(self.h2, self.W3), self.b3)

    def get_cost(self, config):
        l2_loss = tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2) + tf.nn.l2_loss(self.W3)
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true, logits=self.y)) + config.rs * l2_loss
