import tensorflow as tf
Dense = tf.layers.Dense


class CaterpillarNetwork(object):
    """                                  +-
                                         | \
                          x_price o----+-+  |-pred_pct_chg
                                       | | /
                                       | +-
                                       |
      +----+  +----+  +----+        +--+-+
      |    |  |    |  |    |        |    |
      |    +--+    +--+    +-oo  oo-+    |
      +-+--+  +--+-+  +--+-+        +--+-+
        |        |       |             |
        |        |       |             |
        x_1      x_2     x_3           x_n

    It's a caterpillar!
    """
    def __init__(self, dim_hidden=5, dim_response=1, rnn_stack_height=3):
        self.dim_response = dim_response
        self.dim_hidden = dim_hidden
        with tf.variable_scope('rnn'):
            def lstm_cell(nunits):
                return tf.nn.rnn_cell.BasicLSTMCell(self.dim_hidden, activation=tf.tanh)
            self.cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell(dim_hidden) for
                                                        i in range(rnn_stack_height)])
            self.denseout1 = Dense(2 * dim_response, activation=tf.nn.relu)
            self.denseout2 = Dense(dim_response, activation=None)

    def __call__(self, _x_seq, _x_price):
        """
        Parameters
        ----------
        _x: tf.Tensor
            batched input passed to model. _x.shape is (B x T x D).

        Returns
        -------
        tf.Tensor
        """
        _x_seq = tf.transpose(_x_seq, [1, 0, 2])  # to time major
        h_out, _ = tf.nn.dynamic_rnn(cell=self.cell_fw, inputs=_x_seq,
                                     initial_state=None,
                                     time_major=True, dtype=tf.float32)
        seq_out = h_out[-1, :, :]
        # x_concat = tf.concat([seq_out, tf.expand_dims(_x_price, 1)], axis=-1)
        x_concat = seq_out
        return self.denseout2(self.denseout1(x_concat))  # Parameterize logits
