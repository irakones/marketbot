- [Tasks](#org838e9a7)
    - [Detrend](#org5660af9)
    - [Plot output](#org202b101)
    - [Remove periodicity where possible](#org46ff786)
    - [Normalize using training data (?)](#org6159dcc)
  - [Analysis](#orgada9dbd)
    - [setup](#orgfdaba2b)
      - [Load](#orge786748)
      - [model config](#orga38c5b0)
    - [explore](#orge5a50a2)
    - [debug](#org1d403ec)
      - [run](#orge9afa28)
      - [genertor](#org232e9b3)
    - [run](#org9800f36)


<a id="org838e9a7"></a>

# TODO Tasks


<a id="org5660af9"></a>

## Detrend

-   First difference + accumulate differences up to prediction time.


<a id="org202b101"></a>

## Plot output


<a id="org46ff786"></a>

## Remove periodicity where possible


<a id="org6159dcc"></a>

## Normalize using training data (?)


<a id="orgada9dbd"></a>

# Analysis


<a id="orgfdaba2b"></a>

## setup


<a id="orge786748"></a>

### Load

```ipython
import pandas as pd
import os 
from datetime import datetime as dt
df = pd.read_csv('marketbot/data/data.csv')
df['time'] = pd.to_datetime(df.time)
df['timestamp'] = df.time.apply(dt.timestamp)
```

```ipython
from matplotlib import pyplot as plt
import numpy as np
data = df.groupby('timestamp').apply(lambda g: g[['price', 'size']].mean(axis=0))
assert np.diff(data.index).min() > 0  # data is sorted
```


<a id="orga38c5b0"></a>

### model config

```ipython
os.chdir('marketbot/src/catnet')
```

```ipython
from collections import namedtuple
flagdct = {'batch_size': 64,
	   'data_dir': '/tmp/dat/',
	   'hidden_dim': 200,
	   'l1reg_coeff': 1e-10,
	   'l2reg_coeff': 1e-9,
	   # 'l1reg_coeff': 1,
	   # 'l2reg_coeff': 1,
	   'latent_dim': 160,
	   'logdir': '/tmp/log/',
	   'n_epochs': 100000,
	   'n_iterations': 100000,
	   'n_samples_predict': 20,
	   'n_samples_train': 10,
	   'print_every': 1000, 
	   'huber_loss_delta': .1,
	   'use_update_ops': True}  # update_ops control dependency is necessary for batch norm
FLAGS = namedtuple('FLAGS',flagdct.keys())(**flagdct)
ff_params = dict(dim_hidden=20, dim_features=2, rnn_stack_height=3)  
```


<a id="orge5a50a2"></a>

## explore

```ipython
fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10,10))
for i, col in enumerate(data.columns.values):
   data.reset_index().plot.scatter(x='timestamp', y=col, ax=axes[i], s=.7)
```

![img](./obipy-resources/19656F3W.png)

```ipython
from pandas.plotting import scatter_matrix
scatter_matrix(data, alpha=0.2, figsize=(6, 6), diagonal='kde')
```

    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f271599bbe0>,
    <matplotlib.axes._subplots.AxesSubplot object at 0x7f2715aaaeb8>],
    [<matplotlib.axes._subplots.AxesSubplot object at 0x7f27159c8be0>,
    <matplotlib.axes._subplots.AxesSubplot object at 0x7f271582e438>]], dtype=object)

![img](./obipy-resources/19656SBd.png)


<a id="org1d403ec"></a>

## debug


<a id="orge9afa28"></a>

### run

```ipython
import runner
catmodel = runner.Learner(ff_params, FLAGS)
```

```ipython
catmodel.fit(data.values, t_ix=data.index.values)
```

```ipython
from importlib import reload
reload(runner)
```


<a id="org232e9b3"></a>

### genertor

```ipython
import utils
example_generator = utils.GrabSequence(X=data.values, t_ix=data.index.values, input_seq_len=100, time_gap_to_predict=10, stride=1)
```

```ipython
from importlib import reload
reload(utils)
```

```ipython
import tensorflow as tf
tf.reset_default_graph()
g = tf.Graph()
sess = tf.Session(graph=g)
with g.as_default():
    train_ds = tf.data.Dataset.from_generator(example_generator, (tf.float32, tf.float32),
					      (tf.TensorShape([None, 2]), tf.TensorShape([2])))
    train_ds.shuffle(buffer_size=100000)
    train_ds = train_ds.batch(100)
    iterator = tf.data.Iterator.from_structure(train_ds.output_types, train_ds.output_shapes)
    training_init_op = iterator.make_initializer(train_ds)
    batch = iterator.get_next()
```

```ipython
with sess.as_default():
    sess.run(training_init_op)
    while (True):
	b = sess.run(batch)
```


<a id="org9800f36"></a>

## run

```ipython
from runner import Learner
catmodel = Learner(ff_params, FLAGS)
catmodel.fit(data.values, t_ix=data.index.values)
```