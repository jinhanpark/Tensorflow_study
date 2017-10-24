import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

class Trainer:
    def __init__(self, config, model, data, sess):
        self.model = model
        self.data = data
        self.config = config
        self.sess = sess

        self.train_op = tf.train.AdamOptimizer(self.config.lr).minimize(self.model.loss)

        sess.run(tf.global_variables_initializer())

    def train(self):
        for epoch in range(self.config.num_epochs):
            this_loss = 0.
            num_batches = int(self.data.train.num_examples / self.config.batch_size)
            for step in range(num_batches):
                batch = self.data.train.next_batch(self.config.batch_size)
                _, batch_loss = self.sess.run([self.train_op, self.model.loss],
                                              feed_dict={self.model.x: batch[0]})
                this_loss += batch_loss
            if epoch % self.config.display_step == 0:
                print("Epoch: %4d, cost: %f" % (epoch, this_loss))

    def predict(self, num):
        input_img = self.data.test.images[num]
        output_img = self.sess.run(self.model.y,
                                   feed_dict={self.model.x: [input_img]})

        plt.imshow(input_img.reshape((28, 28)), cmap=cm.Greys)
        plt.title("Input Test Image")
        plt.show()

        plt.imshow(output_img.reshape((28, 28)), cmap=cm.Greys)
        plt.title("Reconstructed Image")
        plt.show()

    def get_reconstructed_imgs(self, rand_arr):
        data = self.data.test
        
        input_img = data.images[rand_arr]
        output_img = self.sess.run(self.model.y,
                                   feed_dict={self.model.x: input_img})
      
        return output_img

    def get_random_latent_imgs(self):
        ret = []
        for i in range(16):
            rand_latent = np.random.randn(self.config.bottleneck_size)
            rand_latent[rand_latent<0] = 0
            output_img = self.sess.run(self.model.y_,
                                       feed_dict={self.model.latent: [rand_latent]})
            ret += [output_img]

        return ret

    def get_lin_space_between_two_imgs(self, two_rand):
        ret = []
        data = self.data.test
        bottlenecks = self.sess.run(self.model.bottleneck,
                                    feed_dict={self.model.x: data.images[two_rand]})
        a = bottlenecks[0]
        b = bottlenecks[1]

        for i in range(16):
            new_bottleneck = []
            for j in range(self.config.bottleneck_size):
                new_bottleneck += [a[j] + (b[j] - a[j]) * i / 15]
            out_img = self.sess.run(self.model.y_,
                                    feed_dict={self.model.latent: [new_bottleneck]})
            ret += [out_img]

        return ret
