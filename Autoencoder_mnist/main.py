import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from configs import AEConfig, VAEConfig
from models import Autoencoder, VAE
from trainer import Trainer

from random import randint


def main():
    data = input_data.read_data_sets('/data/mnist/', one_hot=True)

    with tf.Session() as sess:
        print("Autoencoder Training Started\n\n")
        ae_config = AEConfig()
        ae_model = Autoencoder(config)
        ae_trainer = Trainer(ae_config, ae_model, data, sess)
        ae_trainer.train()
        print("Autoencoder Training Ended\n\n")

        # print("VAE Training Started\n\n")
        # vae_config = VAEConfig()
        # vae_model = VAE(vae_config)
        # vae_trainer = Trainer(vae_config, vae_model, data, sess)
        # vae_trainer.train()
        # print("VAE Training Ended\n\n")

        
