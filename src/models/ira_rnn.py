"""
model definition for LSTM
"""
import tensorflow as tf
import numpy as np

ModeKeys = tf.estimator.ModeKeys

DEFAULT_TRAIN_PARAMS = {
    'window': 100,
    'num_epochs': 100,
    'batch_size': 10
}

DEFAULT_MODEL_PARAMS = {
    'stack_size': 3,
    'layer_size': 20,
    'lstm_activation': tf.nn.relu,
    'dense_activation': tf.nn.relu,
    'loss': tf.losses.mean_squared_error
}

MODEL_OUTPUT_DIR = 'trained_models'
def rnn_model_fn(features, labels, mode, params):
    """
    features has shape (batch, seq_length, num_features)
    labels has shape (batch, 1)
    required params:
    layer_size, activation, stack_size, window_length,
    optional params:
    initial weights,
    """

    lstm_layer_fn = lambda: tf.nn.rnn_cell.BasicLSTMCell(
        params['layer_size'],
        activation=params['lstm_activation'])

    lstm_stack = tf.nn.rnn_cell.MultiRNNCell(
        [lstm_layer_fn() for _ in range(params['stack_size'])])

    lstm_out, _ = tf.nn.dynamic_rnn(cell=lstm_stack,
                                    inputs=features,
                                    dtype=tf.float32)

    if mode == ModeKeys.TRAIN:
        dense_input =  tf.squeeze(lstm_out[:, 1, :])
    else:
        dense_input = tf.squeeze(lstm_out)

    predictions = tf.layers.dense(
        inputs=dense_input,
        units=1,
        activation=params['dense_activation'])

    if mode == ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"ages": predictions})


    loss = params['loss'](predictions, labels)
    optimizer = params['optimizer'](learning_rate=params['learning_rate'])
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode,
                                      loss=loss,
                                      train_op=train_op)


def input_fn_wrapper(path, mode, horizon, train_params):
    """
        mode        | output shape
        ------------|-------------
        'train'     | (batch_size, window, feature_dim), (batch_size,)
        'eval'      | (dataset_size, feature_dim)
        'predict'   | (dataset_size, feature_dim)
    """
    data = _read_csv_to_tensor(path)
    if mode == ModeKeys.TRAIN:
        input_fn = _train_input_fn(data, horizon, **train_params)
    elif mode == ModeKeys.EVAL:
        input_fn = _eval_input_fn(data, horizon)
    else:
        input_fn = _predict_input_fn(data)

    return input_fn


def _train_input_fn(data, horizon, num_epochs, batch_size, window):
    targets = _pct_change(data[:, 1], horizon)[window - 1:]
    features = _rolling_windows(data, window)[:-horizon]

    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
    dataset = dataset.shuffle(10000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)

    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next


def _eval_input_fn(data, horizon):
    targets = _pct_change(data[:, 1], horizon)
    return lambda: (tf.reshape(data, [1, -1, 3]), tf.reshape(targets, [1, -1]))

def _predict_input_fn(data):
    return lambda: tf.reshape(data, [1, -1, 3])

def _rolling_windows(data, window):
    return tf.contrib.signal.frame(data, window, 1, axis=0)

def _pct_change(feature_col, horizon):
    return tf.div(feature_col[horizon:], feature_col[: -horizon]) - 1

def _read_csv_to_tensor(path):
    data = np.genfromtxt(path, delimiter=',')[2:, 1:]
    return tf.convert_to_tensor(data, dtype=tf.float32)


def __main__(path='data/clean/data.csv'):
    rnn = tf.estimator.Estimator(model_fn=rnn_model_fn,
                                 params=DEFAULT_MODEL_PARAMS,
                                 model_dir=MODEL_OUTPUT_DIR)

    train_input_fn = input_fn_wrapper(path, 'train', 10, DEFAULT_TRAIN_PARAMS)
    rnn.train(input_fn=train_input_fn)

    eval_input_fn = input_fn_wrapper(path, 'eval', 10, {})
    rnn.evaluate(input_fn=eval_input_fn)

    predict_input_fn = input_fn_wrapper(path, 'infer', 10, {})
    rnn.predict(input_fn=predict_input_fn)

if __name__ == '__main__':
    __main__()
