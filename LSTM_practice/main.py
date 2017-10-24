import tensorflow as tf

from config import Config, GPU_NUMS
from data_loader import proc_raw_data, ProcInput
from model import ProcModel
from train import run_epoch


import os
os.environ['CUDA_VISIBLE_DEVICES'] = GPU_NUMS

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None,
                    "Where the data is stored.")
flags.DEFINE_string("save_path", None,
                    "Model output directory")

FLAGS = flags.FLAGS

def main(_):
  # if not FLAGS.data_path:
  #   raise ValueError("Must set --data_path to data directory")

  config = Config()
  eval_config = Config()

  if FLAGS.data_path:
    config.data_path = FLAGS.data_path
    eval_config.data_path = FLAGS.data_path

  raw_data = proc_raw_data(config)
  train_data, test_data = raw_data
  valid_data = train_data[:160, :]

  with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)

    with tf.name_scope("Train"):
      train_input = ProcInput(data=train_data, config=config, name="TrainInput")
      with tf.variable_scope("Model", reuse=None, initializer=initializer):
        m = ProcModel(is_training=True, config=config, input_=train_input)
      tf.summary.scalar("Training Loss", m.cost)
      tf.summary.scalar("Learning Rate", m.lr)

    with tf.name_scope("Valid"):
      valid_input = ProcInput(data=valid_data, config=config, name="ValidInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mvalid = ProcModel(is_training=False, config=config, input_=valid_input)
      # tf.summary.scalar("Validation Loss", mvalid.cost)

    with tf.name_scope("Test"):
      test_input = ProcInput(config=eval_config, data=test_data, name="TestInput")
      with tf.variable_scope("Model", reuse=True, initializer=initializer):
        mtest = ProcModel(is_training=False, config=eval_config,
                          input_=test_input)

    sv = tf.train.Supervisor(logdir=FLAGS.save_path)
    with sv.managed_session() as session:
      for i in range(config.max_max_epoch):
        lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
        m.assign_lr(session, config.learning_rate * lr_decay)

        print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
        train_entrophy = run_epoch(session, m, eval_op=m.train_op,
                                   verbose=True)
        # train_perplexity = run_epoch(session, m, eval_op=m.train_op,
        #                              verbose=True)
        print("Epoch: %d Train Entrophy: %.3f" % (i + 1, train_entrophy))
        valid_accuracy = run_epoch(session, mvalid, get_acc=True)
        print("Epoch: %d Valid Accuracy: %.3f" % (i + 1, valid_accuracy))

      test_accuracy = run_epoch(session, mtest, get_acc=True)
      print("Test Accuracy: %.3f" % test_accuracy)

      if FLAGS.save_path:
        print("Saving model to %s." % FLAGS.save_path)
        sv.saver.save(session, FLAGS.save_path, global_step=sv.global_step)

if __name__=="__main__":
  tf.app.run()
