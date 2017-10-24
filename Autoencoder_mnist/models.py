import tensorflow as tf

class Autoencoder:
    def __init__(self, config):
        self.x = tf.placeholder(tf.float32, shape=(None, config.input_size))

        W1 = tf.Variable(tf.truncated_normal((config.input_size, config.h1_size),
                                             stddev=0.1),
                         dtype=tf.float32, name="W1")
        b1 = tf.Variable(tf.zeros((1, config.h1_size)),
                         dtype=tf.float32, name="b1")

        h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)

        W2 = tf.Variable(tf.truncated_normal((config.h1_size, config.bottleneck_size),
                                             stddev=0.1),
                         dtype=tf.float32, name="W2")
        b2 = tf.Variable(tf.zeros((1, config.bottleneck_size)),
                         dtype=tf.float32, name="b2")

        self.bottleneck = tf.nn.relu(tf.matmul(h1, W2) + b2)

        W3 = tf.Variable(tf.truncated_normal((config.bottleneck_size, config.h1_size),
                                             stddev=0.1),
                         dtype=tf.float32, name="W3")
        b3 = tf.Variable(tf.zeros((1, config.h1_size)),
                         dtype=tf.float32, name="b3")
        h2 = tf.nn.relu(tf.matmul(self.bottleneck, W3) + b3)

        W4 = tf.Variable(tf.truncated_normal((config.h1_size, config.input_size),
                                             stddev=0.1),
                         dtype=tf.float32, name="W4")
        b4 = tf.Variable(tf.zeros((1, config.input_size)),
                         dtype=tf.float32, name="b4")

        logits = tf.add(tf.matmul(h2, W4), b4)
        self.y = tf.nn.sigmoid(logits)

        

        self.loss = 0.5 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.x, logits=logits))

        self.latent = tf.placeholder(tf.float32, shape=(None, config.bottleneck_size))
        h2_ = tf.nn.relu(tf.matmul(self.latent, W3) + b3)
        logits_ = tf.matmul(h2_, W4) + b4
        self.y_ = tf.nn.sigmoid(logits_)
