"""
    model definition for LSTM
"""
import tensorflow as tf


def model_fn(features, labels, mode, params):
    """
        features has shape (batch, seq_length, num_features)
        labels is batch x 1 dimensional
        required params:
            layer_size, activation, stack_size, window_length,
        optional params:
            initial weights,
    """

    # Function for initalizing lstm layer
    lstm_layer_fn = lambda: tf.nn.rnn_cell.BasicLSTMCell(
        params['layer_size'],
        activation=params['activation'])

    # Make network
    lstm_stack = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_layer_fn() for _ in range(params['stack_size'])])

    if 'initial_state' in params:
        initial_state = params['initial_state']
    else:
        initial_state = None

    feature_seq = tf.split(features, params['window_length'], 1)
    outputs, _ = tf.nn.dynamic_rnn(cell=lstm_stack,
                                   inputs=feature_seq,
                                   initial_state=initial_state)
    


