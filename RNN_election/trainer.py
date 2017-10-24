import numpy as np
import tensorflow as tf

from models import Model

class Trainer():
    def __init__(self, config, data, sess):
        self.model = Model(config, data)

        self.num_epochs = config.num_epochs
        self.display_step = config.display_step

        self.data = data.data

        self.sess = sess
        self.update = tf.train.GradientDescentOptimizer(config.lr).minimize(self.model.cost)
        self.init_state = np.zeros((1, config.state_size))
        tf.global_variables_initializer().run()


    def train(self):
        for step in range(1, self.num_epochs+1):
            _, this_cost = self.sess.run([self.update, self.model.cost], feed_dict={self.model.x: self.data[0], self.model.y_true: self.data[1], self.model.init_state: self.init_state})
            if step % self.display_step == 0:
                print("Step: %4d, cost : %f" % (step, this_cost))


    def predict(self):
        outputs = self.sess.run(self.model.outputs[-1, :], feed_dict={self.model.x: self.data[0], self.model.init_state:self.init_state})
        outputs = np.array(outputs*10000, dtype=int)
        outputs = outputs/100

        name = ["Moon", "Hong", "Ahn", "Yoo", "Shim"]
        
        for i in range(5):
            print(name[i], "may get", outputs[i], "\b%")

