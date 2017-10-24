import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import os

def plot(samples):
  fig = plt.figure(figsize=(4, 4))
  gs = gridspec.GridSpec(4, 4)
  gs.update(wspace=0.05, hspace=0.05)
  for i, sample in enumerate(samples):
    ax = plt.subplot(gs[i])
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(sample.reshape(28, 28), cmap="Greys_r")
  return fig

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def makedir_if_there_is_no(path):
  if not os.path.exists(path):
    os.makedirs(path)
    print("Directory {} was made".format(path))

def grads_clip(grads_and_vars, min_grad, max_grad):
  grads, variables = zip(*grads_and_vars)
  grads = [tf.clip_by_value(grad, min_grad, max_grad) for grad in grads]
  return zip(grads, variables)

def op_with_clipping_for_specific_vars(op, loss, var_lst, min_grad, max_grad):
  grads_and_vars = op.compute_gradients(loss, var_list=var_lst)
  return op.apply_gradients(grads_clip(grads_and_vars, min_grad, max_grad))
