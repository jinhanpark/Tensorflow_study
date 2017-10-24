import numpy as np
import tensorflow as tf
from models import Model

class Trainer():
    def __init__(self, config, data, sess):
        self.model = Model(config, data)

        self.num_epochs = config.num_epochs
        self.display_step = config.display_step

        self.training_data = data.training_data
        self.test_data = data.test_data

        self.sess = sess
        self.update = tf.train.GradientDescentOptimizer(config.lr).minimize(self.model.cost)
        #self.update = tf.train.AdamOptimizer(config.lr).minimize(self.model.cost)

        tf.global_variables_initializer().run()

    def train(self):
        for step in range(1, self.num_epochs+1):
            _, this_cost = self.sess.run([self.update, self.model.cost], feed_dict={self.model.x: self.training_data[0], self.model.y_true: self.training_data[1]})
            if step % self.display_step == 0:
                print("Step: %4d, cost : %f" % (step, this_cost))

    def test(self):
        pred = tf.argmax(self.model.y, 1)
        correctness = tf.equal(pred, tf.argmax(self.model.y_true, 1))
        accuracy = tf.reduce_mean(tf.cast(correctness, tf.float32))

        training_accuracy = self.sess.run(accuracy, feed_dict={self.model.x: self.training_data[0], self.model.y_true: self.training_data[1]})
        test_accuracy = self.sess.run(accuracy, feed_dict={self.model.x: self.test_data[0], self.model.y_true: self.test_data[1]})

        print("Training Accuracy:", training_accuracy, "\nTest Accuracy:", test_accuracy)
