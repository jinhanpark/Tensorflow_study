import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer as xavier_init
import numpy as np

import matplotlib.pyplot as plt
import os
from utils import *


class GAN:
  def __init__(self, sess, config, data):
    self.config = config
    self.sess = sess
    self.data = data
    self.build_model()
  
  def build_model(self):
    self.inputs = tf.placeholder(tf.float32, [None, self.config.out_dim])
    self.z = tf.placeholder(tf.float32, [None, self.config.z_dim])

    self.G = self.generator(self.z)
    self.D, self.D_logits = self.discriminator(self.inputs)
    self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.d_loss_real = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_logits, labels=tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
    self.d_loss = self.d_loss_real + self.d_loss_fake
    
    self.g_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
        logits=self.D_logits_, labels=tf.ones_like(self.D_)))

    min_grad = -1e-2
    max_grad = 1e-2
    self.train_op = tf.train.AdamOptimizer(self.config.lr)
    self.d_opt = op_with_clipping_for_specific_vars(
      self.train_op, self.d_loss, self.d_vars, min_grad, max_grad)
    self.g_opt = op_with_clipping_for_specific_vars(
      self.train_op, self.g_loss, self.g_vars, min_grad, max_grad)
    
    self.sess.run(tf.global_variables_initializer())    
    self.saver = tf.train.Saver()

  def train(self):
    makedir_if_there_is_no(self.config.out_dir)
    makedir_if_there_is_no(self.config.ckpt_dir)
    
    counter = 0
    if self.config.load_checkpoint:
      load_success, ckpt_cnt = self.load()
      if load_success:
        counter = ckpt_cnt
        print("Checkpoint Loaded.")
      else:
        print("!!!!FAILED to load checkpoint!!!!")
    
    num_batches = self.data.train.num_examples // self.config.batch_size
    for epoch in range(self.config.num_epochs):
      epoch_d_loss = 0.
      epoch_g_loss = 0.
      for i in range(num_batches):
        batch_x, _ = self.data.train.next_batch(self.config.batch_size)
        _, batch_d_loss = self.sess.run(
          [self.d_opt, self.d_loss],
          feed_dict={self.inputs: batch_x,
                     self.z: self.sample_z(self.config.batch_size)})
        epoch_d_loss += batch_d_loss
        
        _, batch_g_loss = self.sess.run(
          [self.g_opt, self.g_loss],
          feed_dict={self.z: self.sample_z(self.config.batch_size)})
        epoch_g_loss += batch_g_loss

      if counter % self.config.display_step == 0:
        print("Epoch: %7d, D loss: %.4f, G loss: %.4f" % (counter, epoch_d_loss, epoch_g_loss))
      
      if counter % self.config.save_step == 0:
        self.save(counter)
        
        samples = self.generate(16)
        fig = plot(samples)
        out_dir = os.path.join(self.config.out_dir,
                               "%05d.png"%counter)
        plt.savefig(out_dir, bbox_inches='tight')
        plt.close(fig)
      counter += 1

  def generator(self, z):
    with tf.variable_scope("generator") as scope:
      g_w1 = tf.get_variable(
        "g_w1", [self.config.z_dim, self.config.h_dim],
        initializer=xavier_init())
      g_b1 = tf.get_variable(
        "g_b1", [self.config.h_dim],
        initializer=tf.zeros_initializer())
      g_w2 = tf.get_variable(
        "g_w2", [self.config.h_dim, self.config.out_dim],
        initializer=xavier_init())
      g_b2 = tf.get_variable(
        "g_b2", [self.config.out_dim],
        initializer=tf.zeros_initializer())

      h = tf.nn.relu(tf.matmul(z, g_w1) + g_b1)
      return tf.nn.sigmoid(tf.matmul(h, g_w2) + g_b2)

  def discriminator(self, x, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
      d_w1 = tf.get_variable(
        'd_w1', [self.config.out_dim, self.config.h_dim],
        initializer=xavier_init())
      d_b1 = tf.get_variable(
        'd_b1', [self.config.h_dim],
        initializer=tf.zeros_initializer())
      d_w2 = tf.get_variable(
        'd_w2', [self.config.h_dim, 1],
        initializer=xavier_init())
      d_b2 = tf.get_variable(
        'd_b2', [1],
        initializer=tf.zeros_initializer())
      
    h = tf.nn.relu(tf.matmul(x, d_w1) + d_b1)
    logits = tf.matmul(h, d_w2) + d_b2
    return tf.nn.sigmoid(logits), logits

  def sample_z(self, n):
    return np.random.uniform(-1., 1., size=[n, self.config.z_dim])

  def generate(self, n):
    return self.sess.run(self.G, feed_dict={self.z: self.sample_z(n)})

  def save(self, step):
    model_name = "GAN.model"
    ckpt_dir = os.path.join(self.config.ckpt_dir, model_name)
    a = self.saver.save(self.sess, ckpt_dir, global_step=step)

  def load(self):
    import re
    print('Trying to load checkpoint')
    ckpt = tf.train.get_checkpoint_state(self.config.ckpt_dir)

    if ckpt and ckpt.model_checkpoint_path:
      ckpt_path = ckpt.model_checkpoint_path
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_path)).group(0))
      self.saver.restore(self.sess, ckpt_path)
      return True, counter
    else:
      return False, 0
