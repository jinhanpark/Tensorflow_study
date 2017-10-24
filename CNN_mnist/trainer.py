import numpy as np
import tensorflow as tf
from models import Model

class Trainer():
    def __init__(self, config, data, sess):
        self.model = Model(config, data)

        self.total_step = config.total_step
        self.batch_size = config.batch_size
        self.display_step = config.display_step

        self.data = data
        
        self.sess = sess

        with tf.name_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(config.lr).minimize(self.model.loss)
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(self.model.logits, 1), tf.argmax(self.model.y_, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        sess.run(tf.global_variables_initializer())
        
    def train(self):
        for i in range(self.total_step):
            batch = self.data.train.next_batch(self.batch_size)
            if i%self.display_step == 0:
                train_accuracy = self.accuracy.eval(feed_dict={
                    self.model.x:batch[0], self.model.y_: batch[1], self.model.keep_prob: 1.0})
                print("step %d, training accuracy : %g"%(i, train_accuracy))
            self.train_op.run(feed_dict={self.model.x: batch[0], self.model.y_: batch[1], self.model.keep_prob: 0.5})
        
    def test(self):
        acc_vector = []
        for i in range(10):
            batch = [self.data.test.images[i*1000:(i+1)*1000],
                     self.data.test.labels[i*1000:(i+1)*1000]]
            acc_vector.append(self.accuracy.eval(feed_dict={
                self.model.x: batch[0], self.model.y_: batch[1], self.model.keep_prob: 1.0}))
        print("Test Accuracy : %g"%np.mean(acc_vector))

    def validate(self):
        acc_vector = []
        for i in range(5):
            batch = [self.data.validation.images[i*1000:(i+1)*1000],
                     self.data.validation.labels[i*1000:(i+1)*1000]]
            acc_vector.append(self.accuracy.eval(feed_dict={
                self.model.x: batch[0], self.model.y_: batch[1], self.model.keep_prob: 1.0}))
        print("Validation Accuracy : %g"%np.mean(acc_vector))
