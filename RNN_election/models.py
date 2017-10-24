import numpy as np
import tensorflow as tf

class Model():
    def __init__(self, config, data):
        self.seq_len = data.data[0].shape[0]
        self.x = tf.placeholder(tf.float32, (self.seq_len, config.fan_in))
        self.y_true = tf.placeholder(tf.float32, (self.seq_len-1, config.fan_out))
        self.init_state = tf.placeholder(tf.float32, (1, config.state_size))

        self.get_variables(config)

        self.get_graph(self.seq_len)

        self.get_cost()

    def get_variables(self, config):
        self.Wxh = tf.Variable(tf.truncated_normal((config.fan_in, config.state_size), stddev=0.01, dtype=tf.float32, name="Wxh"))
        self.Whh = tf.Variable(tf.truncated_normal((config.state_size, config.state_size), stddev=0.01, dtype=tf.float32, name="Wxx"))
        self.Why = tf.Variable(tf.truncated_normal((config.state_size, config.fan_out), stddev=0.01, dtype=tf.float32, name="Why"))
        self.bh = tf.Variable(tf.zeros((1, config.state_size)), dtype=tf.float32, name="bh")
        self.by = tf.Variable(tf.zeros((1, config.fan_out)), dtype=tf.float32, name="by")

    def get_graph(self, seq_len):
        self.state = self.init_state
        self.outputs = []
        for i in range(seq_len):
            self.state = tf.tanh(tf.matmul([self.x[i, :]], self.Wxh) + tf.matmul(self.state, self.Whh) + self.bh)
            self.logit = tf.matmul(self.state, self.Why) + self.by
            self.y = tf.nn.softmax(self.logit)
            self.outputs.append(self.y)
        self.outputs = tf.concat(self.outputs, 0)

    def get_cost(self):
        self.cost = tf.nn.l2_loss(self.y_true - self.outputs[:-1, :])
