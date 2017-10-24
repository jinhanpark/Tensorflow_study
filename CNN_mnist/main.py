from config import Config
from data_loader import mnist, show_pics
from trainer import Trainer


import numpy as np
import tensorflow as tf


def main():
    for i in [17, 28, 52]:
        show_pics(mnist, i)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    
    with tf.Session(config=sess_config) as sess:
        config = Config()
        data = mnist
        trainer = Trainer(config, data, sess)
        writer = tf.summary.FileWriter(config.log_dir, sess.graph)
        #tensorboard --logdir=./

        trainer.train()
        trainer.test()
        trainer.validate()

if __name__=="__main__":
    main()
