import tensorflow as tf

class ProcModel:
  def __init__(self, is_training, config, input_):
    self._input = inputs = input_

    batch_size = inputs.batch_size
    num_steps = config.num_steps
    num_class = config.num_class
    size = config.hidden_size

    def lstm_cell():
      return tf.contrib.rnn.BasicLSTMCell(
        size, forget_bias=0.0, state_is_tuple=True,
        reuse=tf.get_variable_scope().reuse
      )

    attn_cell = lstm_cell
    if is_training and config.keep_prob < 1:
      def attn_cell():
        return tf.contrib.rnn.DropoutWrapper(
          lstm_cell(), output_keep_prob=config.keep_prob
        )
    cell = tf.contrib.rnn.MultiRNNCell(
      [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True
    )

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    self.input_place = tf.placeholder(tf.float32, shape=[batch_size, num_steps])
    self.target_place = tf.placeholder(tf.int32, shape=[batch_size, num_steps])
    outputs = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(tf.expand_dims(self.input_place[:, time_step], 1), state)
        outputs.append(cell_output)

    output = tf.reshape(tf.stack(axis=1, values=outputs), [-1, size])
    softmax_w = tf.get_variable(
      "softmax_w", [size, num_class], dtype=tf.float32
    )
    softmax_b = tf.get_variable("softmax_b", [num_class], dtype=tf.float32)
    logits = tf.matmul(output, softmax_w) + softmax_b

    logits = tf.reshape(logits, [batch_size, num_steps, num_class])
    last_logits = logits[:, -1, :]
    self.pred = tf.argmax(last_logits, axis=1)

    one_hot_label = tf.one_hot(self.target_place[:, -1], 9, on_value=1.0, off_value=0.0, axis=-1)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_label, logits=last_logits)
    self._cost = cost = tf.reduce_sum(loss)

    # loss = tf.contrib.seq2seq.sequence_loss(
    #   logits,
    #   self.target_place,
    #   tf.ones([batch_size, num_steps], dtype=tf.float32),
    #   average_across_timesteps=False,
    #   average_across_batch=True
    # )

    # self._cost = cost = tf.reduce_sum(loss)
    self._final_state = state

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self._lr)
    self._train_op = optimizer.apply_gradients(
      zip(grads, tvars),
      global_step=tf.contrib.framework.get_or_create_global_step()
    )

    self._new_lr = tf.placeholder(
      tf.float32, shape=[], name="new_learning_rate"
    )
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr : lr_value})

  def get_next_batch(self):
    return self._input.get_next_batch()

  @property
  def input(self):
    return self._input

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op
