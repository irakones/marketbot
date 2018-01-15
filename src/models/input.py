import tensorflow as tf
import numpy as np 


def train_input_fn(path, window, horizon):
    data = np.genfromtxt(path, delimter=',')[2:, 1]
    data = tf.convert_to_tensor(data)
    targets = data[horizon + window - 1 :, 1] / data[window - 1 : -horizon, 1] - 1
    features = tf.contrib.signal.frame(data, window, 1, axis=0)[:-horizon]
