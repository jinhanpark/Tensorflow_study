import tensorflow as tf

from config import Config
from tensorflow.examples.tutorials.mnist import input_data
from model import GAN
from utils import *

def main():
  sess = tf.Session()
  config = Config()
  data = input_data.read_data_sets(config.data_dir, one_hot=True)
  gan = GAN(sess, config, data)

  show_all_variables()
  
  gan.train()

if __name__=="__main__":
  main()
