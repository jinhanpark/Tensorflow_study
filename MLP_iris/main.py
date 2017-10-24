import numpy as np
import tensorflow as tf

from trainer import Trainer
from data_loader import GetIrisData
from config import Config


def main():
    data_path = 'iris.data'

    with tf.Session() as sess:

        config = Config()
        data = GetIrisData(data_path)
        trainer = Trainer(config, data, sess)

        trainer.train()
        trainer.test()

if __name__=="__main__":
    main()
