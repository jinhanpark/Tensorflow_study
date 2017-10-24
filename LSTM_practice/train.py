import numpy as np
import time

def run_epoch(sess, model, eval_op=None, verbose=False, get_acc=False):
  start_time = time.time()
  costs = 0.0
  iters = 0
  cost_denominator = 0
  acc_list = []

  for batch_step in range(model.input.num_batches):
    state = sess.run(model.initial_state)
    fetches = {
      "cost": model.cost,
      "final_state": model.final_state,
    }
    if eval_op is not None:
      fetches["eval_op"] = eval_op

    for epoch_step in range(model.input.epoch_size):
      # print("Epoch Step %d is in progress"%epoch_step)

      this_input = model.input.inputs[batch_step][epoch_step]
      this_target = model.input.targets[batch_step][epoch_step]
      feed_dict = {model.input_place:this_input, model.target_place:this_target}
      for i, (c, h) in enumerate(model.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h

      vals = sess.run(fetches, feed_dict)
      cost = vals["cost"]
      state = vals["final_state"]

      costs += cost
      cost_denominator += model.input.batch_size
      iters += model.input.num_steps

      display_step = model.input.epoch_size // 5
      if verbose and epoch_step % display_step == display_step - 1:
        print("%.3f loss: %.3f speed: %.0f day per sec" %
              (epoch_step * 1.0 / model.input.epoch_size, costs / cost_denominator,
               iters * model.input.batch_size / (time.time() - start_time)))
        # print("%.3f perplexity: %.3f speed: %.0f day per sec" %
        #       (epoch_step * 1.0 / model.input.epoch_size, np.exp(costs / iters),
        #        iters * model.input.batch_size / (time.time() - start_time)))
    if get_acc:
      batch_pred = sess.run(model.pred, feed_dict)
      print(batch_pred)
      batch_true = this_target[:, -1]
      print(batch_true)
      batch_correctness = np.equal(batch_pred, batch_true)
      batch_acc = np.mean(batch_correctness)
      acc_list.append(batch_acc)
    progress = (batch_step + 1) / model.input.num_batches * 100
    print("epoch progress : %f %%"%progress)
  if get_acc:
    accuracy = np.mean(acc_list)
    return accuracy
  else:
    return costs / cost_denominator
    # return np.exp(costs / iters)
