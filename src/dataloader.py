from collections import deque
import pandas as pd
import numpy as np

def get_data(path, interval_length, predict_length):
    """ interval_length: number of seconds to aggregate together to regularize
                         sample frequency
        window_length: length to be used to rolling window generator
        stride_length: step size to increment windows
        predict_length: num# intervals ahead to predict price (e.g., if
                        interval_length is 1 and predict_length is 10, then at time t we are
                        predicting price at time t + 10 """

    df = pd.read_csv(path, index_col='sequence')
    df.time = pd.to_datetime(df.time)
    agg_df = discretize(df, interval_length)

    agg_df.price = agg_df.price.fillna(method='ffill')
    agg_df['change'] = agg_df.price.pct_change()

    outcomes  = agg_df.price.pct_change(periods=predict_length).as_matrix()[11:]
    features = agg_df[['volume', 'price', 'change']].as_matrix()[1:]

    return features, outcomes

def discretize(df, interval_length):
    """ interval_length = number of seconds to aggregate in """
    if interval_length == 0:
        grouped = df.groupby('time')
    else:
        grouped = df.resample('{}S'.format(interval_length), on='time')
    return grouped.apply(mean_price).reset_index()

def mean_price(g):
    """ computes size-weighted price of trades """
    total_volume = g.volume.sum()
    if total_volume:
        price = (g.volume * g.price).sum() / total_volume 
    else:
        price = None
    return pd.Series([total_volume, price], ['volume', 'price'])

def window_gen(features, outcomes, window_length, predict_length):
    """ creates a generator for rolling window of features """

    cur = deque(features[:window_length, [0, 2]])
    last_price = cur[-1][1] 

    for i in range(window_length, len(features) - predict_length):
        yield [cur, outcomes[i], last_price]
        cur.popleft()
        cur.append(features[i, [0, 2]])
        last_price = features[i, 1]

def main(path='data/data.csv', predict_length=10, interval_length=1, window_length=100):
    features, outcomes = get_data(path, interval_length, predict_length)
    return window_gen(features, quantize(outcomes), window_length, predict_length)

def quantize(items, amin=-0.01, amax=0.01, step=1e-5):
    return np.digitize(np.clip(items, amin, amax), np.arange(amin, amax, step))
