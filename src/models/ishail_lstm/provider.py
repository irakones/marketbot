import pandas as pd
import tensorflow as tf


class (object):
    """
    Generate next sequence based on desired stride.
    """
    def __init__(self, X, t_ix, input_seq_len, time_gap_to_predict, stride=1):
        """
        Parameters
        ----------
        X: numpy.array
            (T x 2) feature array
        t_ix: numpy.array
            (T x 1) index array
        input_seq_len: int
            Input sequnce length
        time_gap_to_predict: float
            time to prediction, in seconds.
        stride: int
            how many observations to increment by before taking next sequence sample.
        """
        self.stride = stride
        self.data = pd.DataFrame(X, index=t_ix)
        self.input_seq_len = input_seq_len
        self.time_gap_to_predict = time_gap_to_predict
        self.get_end_ptr = lambda start: start + self.input_seq_len
        self.get_predict_time = lambda start: (self.data.index[self.get_end_ptr(start)] +
                                               self.time_gap_to_predict)
        self.latest_time = self.data.index[-1]
        # TODO: this seems like an awful way to do this
        self.get_predict_index = lambda start: \
            (None if (self.get_predict_time(start) > self.latest_time)
             else self.data.index[self.data.index > self.get_predict_time(start)][0])

    def __call__(self):
        

        self.start_ptr = 0
        predict_index = self.get_predict_index(self.start_ptr)
        while predict_index:
            yield (self.data.iloc[self.start_ptr:self.start_ptr + self.input_seq_len].values,
                   self.data.loc[predict_index, :].values)
            self.start_ptr += self.stride
            predict_index = self.get_predict_index(self.start_ptr)


class SequenceGenerator(object):
    """
    Generate next sequence based on desired stride.
    """
    def __init__(self, X, t_ix, input_seq_len, time_gap_to_predict, stride=1):
        """
        Parameters
        ----------
        X: numpy.array
            (T x 2) feature array
        t_ix: numpy.array
            (T x 1) index array
        input_seq_len: int
            Input sequnce length
        time_gap_to_predict: float
            time to prediction, in seconds.
        stride: int
            how many observations to increment by before taking next sequence sample.
        """
        self.stride = stride
        self.data = pd.DataFrame(X, index=t_ix)
        self.input_seq_len = input_seq_len
        self.time_gap_to_predict = time_gap_to_predict
        self.get_end_ptr = lambda start: start + self.input_seq_len
        self.get_predict_time = lambda start: (self.data.index[self.get_end_ptr(start)] +
                                               self.time_gap_to_predict)
        self.latest_time = self.data.index[-1]
        # TODO: this seems like an awful way to do this
        self.get_predict_index = lambda start: \
            (None if (self.get_predict_time(start) > self.latest_time)
             else self.data.index[self.data.index > self.get_predict_time(start)][0])

    def __call__(self):
        self.start_ptr = 0
        predict_index = self.get_predict_index(self.start_ptr)
        while predict_index:
            yield (self.data.iloc[self.start_ptr:self.start_ptr + self.input_seq_len].values,
                   self.data.loc[predict_index, :].values)
            self.start_ptr += self.stride
            predict_index = self.get_predict_index(self.start_ptr)
