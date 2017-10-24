import tensorflow as tf

from trainer import Trainer
from data_loader import GetData
from config import Config


def main():
    data_path = 'Gallup.xlsx'

    with tf.Session() as sess:
        
        config = Config()
        data = GetData(data_path)
        trainer = Trainer(config, data, sess)

        trainer.train()

        trainer.predict()

if __name__=="__main__":
    main()
