GPU_NUMS = "3"

class Config(object):
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_steps = 100
  hidden_size = 1500
  max_epoch = 500
  max_max_epoch = 1500
  keep_prob = 0.35
  lr_decay = 0.9
  batch_size = 160

  lower_bounds = [-30, -15, -5, -2, -1, 1, 2, 5, 15]
  num_class = len(lower_bounds)

  train_ratio = 0.8
  test_ratio = 0.2

  proc_gap = 1
  time_length = 1000

  data_path = "/data/datasets/proc_data"
  #data_path = "d:/datasets/proc_data"
