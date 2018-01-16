import tensorflow as tf
import numpy as np
from .utils import GrabSequence
from .feedforward import CaterpillarNetwork


class Learner(object):
    """
    RNN to n second ahead prediction
    """
    def __init__(self, ff_params, FLAGS, name='CaterpillarNet'):
        """
        Parameters
        ----------
        ff_params: dict
            feedforward parameters
        FLAGS: dict
            flag dictionary
        name: str
            model name
        """
        self.ff_params = ff_params
        self.FLAGS = FLAGS
        self.name = name

    def fit(self, X, y, t_ix=None, input_seq_len=100, tdelta_predict=10, stride=1):
        """
        Parameters
        ----------
        X : pd.DataFrame
            indexed by Timestamp (sorted). Columns are average_price, average_volume observations.
        y : pd.Series
            indexed by Timestamp (sorted). Columns are average_price, average_volume observations.

        input_seq_len: int
            length of RNN
        tdelta_predict: float
            predict price / volume at tdelta + timestamp latest obsevation
        stride: int
            Stride at which to sample.
        """
        if t_ix is None:
            t_ix = np.arange(X.shape[0])
        example_generator = GrabSequence(X, t_ix, input_seq_len, tdelta_predict, stride=stride)
        self.initialize_train_graph(example_generator)
        self.train(self.FLAGS.n_epochs)

    def initialize_train_graph(self, example_generator):
        tf.reset_default_graph()
        self.g = tf.Graph()
        self.sess = tf.Session(graph=self.g)
        with self.g.as_default():
            self.feedforward = CaterpillarNetwork(**self.ff_params,
                                                  dim_response=(example_generator.dim_Y *
                                                                example_generator.Y_n_categories))
        dim_X = example_generator.dim_X
        dim_Y = example_generator.dim_Y
        Y_n_categories = example_generator.Y_n_categories
        with self.g.as_default():
            with tf.variable_scope(self.name):
                train_ds = tf.data.Dataset.from_generator(example_generator,
                                                          (tf.float32, tf.float32, tf.int32),
                                                          (tf.TensorShape([None, dim_X]),
                                                           tf.TensorShape([]),
                                                           tf.TensorShape([dim_Y])))
                train_ds.shuffle(buffer_size=100000)
                train_ds = train_ds.batch(self.FLAGS.batch_size)
                iterator = tf.data.Iterator.from_structure(train_ds.output_types,
                                                           train_ds.output_shapes)
                self.training_init_op = iterator.make_initializer(train_ds)
                global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = tf.Variable(.001, trainable=False)
                self.starter_learning_rate = starter_learning_rate
                lr_decay = tf.Variable(0.999995, trainable=False)
                decay_steps = tf.Variable(1, trainable=False)
                learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                           decay_steps, lr_decay, staircase=True)
                batch = iterator.get_next()
                self.X_seq = batch[0]
                self.X_price = batch[1]
                self.y = batch[2]
                yhat_pre = self.feedforward(self.X_seq, self.X_price)
                self.yhat = tf.reshape(yhat_pre, [-1, dim_Y, Y_n_categories])
                self.yhat_softmax = tf.nn.softmax(self.yhat)
                with tf.variable_scope('loss'):
                    ce_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y,
                                                                     logits=self.yhat,
                                                                     reduction='weighted_mean')
                    tf.summary.scalar('cross_ent_loss', ce_loss)
                    self.ce_loss = ce_loss
                    w_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                    w_ss_list = [tf.nn.l2_loss(v) for v in w_vars if 'bias' not in v.name]
                    l2reg = tf.reduce_sum(w_ss_list, name='l2reg')  # l2 regularization
                    self.l2reg = l2reg
                    w_l1_list = [tf.reduce_sum(tf.abs(v)) for v in w_vars if 'bias' not in
                                 v.name]
                    l1reg = tf.reduce_sum(w_l1_list, name='l1reg')
                    self.l1reg = l1reg
                    reg_loss = self.FLAGS.l2reg_coeff * l2reg + self.FLAGS.l1reg_coeff * l1reg
                    self.reg_loss_sy = reg_loss
                    total_loss_op = tf.add(ce_loss, reg_loss, name='loss_op')
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.variable_scope(self.name):
                with tf.variable_scope('optimization'):
                    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                    self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    if self.FLAGS.use_update_ops:
                        with tf.control_dependencies(self.update_ops):
                            self.train_op = optimizer.minimize(total_loss_op)
                    else:
                        self.train_op = optimizer.minimize(total_loss_op)

                self.init = tf.group(tf.global_variables_initializer(),
                                     tf.local_variables_initializer())
                self.summary_merged = tf.summary.merge_all()
                self.summary_writer = tf.summary.FileWriter(self.FLAGS.logdir, self.g)

                self.i_e = 0

    def train(self, epochs=5):
        """
        Parameters
        ----------
        epochs: int
            Number of epochs for training description.
        """
        with self.g.as_default():
            sess = self.sess
            starter_learning_rate = self.starter_learning_rate
            training_init_op = self.training_init_op
            update_ops = self.update_ops if (len(self.update_ops) > 0) else tf.ones(1)
            train_op = self.train_op
            init = self.init
            l1reg = self.l1reg
            l2reg = self.l2reg
            with sess.as_default():
                # Run the initializer
                if self.i_e == 0:
                    print('initializing weights')
                    sess.run(init)
                i_e_start = self.i_e + 1
                sess.run(tf.assign(starter_learning_rate, 1e-3))
                for i_e in range(i_e_start, epochs):
                    sess.run(training_init_op)
                    try:
                        # for i_es in range(epoch_steps):
                        i_es = 0
                        while(True):
                            try:
                                if (i_es > 0) or (self.summary_merged is None):
                                    if (((i_es % 10) == 0) and ((i_e % 5) == 0)):
                                        _, _, l1r, l2r, cross_ent_loss, reg_loss = \
                                            sess.run([update_ops, train_op, l1reg, l2reg,
                                                      self.ce_loss, self.reg_loss_sy])
                                        print((l1r, l2r, cross_ent_loss, reg_loss))
                                    else:
                                        sess.run([update_ops, train_op])
                                else:
                                    _, _, summary, l1r, l2r = \
                                        sess.run([update_ops, train_op, self.summary_merged, l1reg,
                                                  l2reg])
                                    self.summary_writer.add_summary(summary, i_e)
                                i_es += 1
                            except tf.errors.OutOfRangeError as inst:
                                # Will raise exception if number of batches is exceeded.
                                print("crashed on iteration {}".format(i_es))
                                raise Exception("crashed on iteration {}".format(i_es))
                    except Exception as inst:
                        print(inst)
                    self.i_e = i_e

    def predict(self, x):
        """
        Evaluate prediction over model.

        Parameters
        ----------
        x: numpy.array
        """
        feed_dict = {self.x: np.expand_dims(x, axis=[0, 1])}
        return self.sess.run(self.yhat_softmax, feed_dict)
