"""
    Load timeseries data for RNN
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.getcwd(), 'data')
DECIMALS = 8

def clean_data(filename, interval_length=1):
    raw_path = os.path.join(DATA_DIR, 'raw/{}'.format(filename))
    df = pd.read_csv(raw_path, index_col='sequence')

    df.time = pd.to_datetime(df.time)
    agg_df = _discretize(df, interval_length)
    agg_df.time = agg_df.time.astype(np.int64) // 10 ** 9
    agg_df.price = agg_df.price.fillna(method='ffill')
    agg_df['change'] = agg_df.price.pct_change()
    agg_df = agg_df.round(DECIMALS)

    clean_path = os.path.join(DATA_DIR, 'clean/{}'.format(filename))
    agg_df.to_csv(clean_path, index=False)


def _discretize(df, interval_length):
    if interval_length == 0:
        grouped = df.groupby('time')
    else:
        grouped = df.resample('{}S'.format(interval_length), on='time')
    return grouped.apply(_group_fn).reset_index()


def _group_fn(group):
    total_volume = group.volume.sum()
    if total_volume:
        price = (group.volume * group.price).sum() / total_volume
    else:
        price = None
    return pd.Series([total_volume, price], ['volume', 'price'])
