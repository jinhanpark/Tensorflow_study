import os

import pandas as pd
import numpy as np
import tensorflow as tf


def _make_csv_to_arr(file_path):
  df = pd.read_csv(file_path, header=0, index_col="Date", verbose=False)
  arr = np.array(df, dtype=np.float32)
  arr = np.transpose(arr)
  return arr[:, ::-1]

def proc_raw_data(config):
  data_path = config.data_path
  train_path = os.path.join(data_path, "proc_train.csv")
  test_path = os.path.join(data_path, "proc_test.csv")
  
  train_data = _make_csv_to_arr(train_path)
  test_data = _make_csv_to_arr(test_path)
  
  return train_data, test_data

def divide_class(arr, lower_bounds):
  result = np.zeros_like(arr, dtype=np.int32)
  for i in range(len(lower_bounds)):
    result[arr > lower_bounds[i]] = i
  return result

def truncated_input_producer(raw_data, config, batch_step=0, name=None):
  batch_size = config.batch_size
  num_steps = config.num_steps
  with tf.name_scope(name, "InputProducer", [raw_data, batch_size, num_steps]):
    data = raw_data[:, :-1]
    targets = divide_class(raw_data[:, 1:], config.lower_bounds)
    batch_len = data.shape[1]
    epoch_size = batch_len // num_steps
    
    xs = []
    ys = []
    
    for i in range(epoch_size):
      x = data[batch_step * batch_size : (batch_step + 1) * batch_size,
               i * num_steps : (i + 1) * num_steps]
      y = targets[batch_step * batch_size : (batch_step + 1) * batch_size,
                  i * num_steps : (i + 1) * num_steps]
      xs.append(x)
      ys.append(y)
    return xs, ys

    # assertion = tf.assert_positive(
    #   epoch_size,
    #   message="epoch_size == 0, decrease batch_size or num_steps"
    # )
    # with tf.control_dependencies([assertion]):
    #   epoch_size = tf.identity(epoch_size, name="epoch_size")

    # i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
    # x = tf.strided_slice(data, [batch_step * batch_size, i * num_steps],
    #                      [(batch_step + 1) * batch_size, (i + 1) * num_steps])
    # x.set_shape([batch_size, num_steps])
    # y = tf.strided_slice(targets, [batch_step * batch_size, i * num_steps],
    #                      [(batch_step + 1) * batch_size, (i + 1) * num_steps])
    # y.set_shape([batch_size, num_steps])
    # return x, y

class ProcInput:
  def __init__(self, data, config, name=None):
    self.data = data
    self.config = config
    self.name = name

    self.batch_len = batch_len = data.shape[1]
    self.batch_size = batch_size = config.batch_size
    self.num_batches = (len(data) // batch_size)
    self.num_steps = num_steps = config.num_steps
    self.epoch_size = batch_len // num_steps

    self.inputs = []
    self.targets = []
    self.get_batch_lists()

  def get_batch_lists(self):
    for step in range(self.num_batches):
      input_data, target = truncated_input_producer(
        self.data, self.config, batch_step=step, name=self.name
      )
      self.inputs.append(input_data)
      self.targets.append(target)
    print("Batch lists generated.")
