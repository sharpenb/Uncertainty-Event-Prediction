import tensorflow as tf

class BaseModel:
    def __init__(self, n_classes, rnn_hidden_dim, mark_emb_dim, lr, regularization):
        self.n_classes = n_classes
        self.rnn_hidden_dim = rnn_hidden_dim
        self.mark_emb_dim = mark_emb_dim
        self.lr = lr
        self.regularization = regularization

        # Define input placeholders
        self.x = tf.placeholder(tf.int32, [None, None], 'x')
        self.tx = tf.placeholder(tf.float32, [None, None], 'tx')
        self.y = tf.placeholder(tf.int32, [None, None], 'y')
        self.ty = tf.placeholder(tf.float32, [None, None], 'ty')
        self.s = tf.placeholder(tf.int32, [None], 'sequence-length')

        # Define mark embedding matrix
        self.mark_emb_weight = tf.get_variable('mark-emb-matrix', [n_classes, mark_emb_dim])

    def rnn(self, input, seq_len, reuse=None):
        cell = tf.nn.rnn_cell.GRUCell(self.rnn_hidden_dim, reuse=reuse)
        h, _ = tf.nn.dynamic_rnn(cell, input, sequence_length=seq_len, dtype=tf.float32)
        return h

    def mark_embedding(self, x):
        return tf.nn.embedding_lookup(self.mark_emb_weight, x)

    def loss_regularization(self):
        var = tf.trainable_variables()
        return tf.add_n([tf.nn.l2_loss(v) for v in var if 'bias' not in v.name]) * self.regularization

    def optimize_step(self, loss):
        optimizer = tf.train.AdamOptimizer(self.lr)
        gradients, variables = zip(*optimizer.compute_gradients(loss))
        # gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimize = optimizer.apply_gradients(zip(gradients, variables))
        return optimize

    def get_accuracy(self):
        equal = tf.cast(tf.equal(tf.argmax(self.logits, -1, output_type=tf.int32), self.y), tf.float32)
        return self.aggregate(equal, self.s)

    def aggregate(self, values, lengths):
        """Calculate masked average of values.

        Arguments:
            values (Tensor): shape (batch size, sequence length)
            lengths (Tensor): shape (batch size)

        Returns:
            mean (float): Average value in values taking padding into account
        """
        values = tf.where(tf.is_nan(values), tf.zeros_like(values), values) # Remove nans
        mask = tf.cast(tf.sequence_mask(lengths, tf.shape(values)[1]), tf.float32)
        values *= mask
        return tf.reduce_sum(values) / tf.cast(tf.reduce_sum(lengths), tf.float32)
