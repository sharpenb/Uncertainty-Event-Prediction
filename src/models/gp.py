import math
import numpy as np
import tensorflow as tf

from .base import BaseModel

def get_shape(x):
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    return [static[i] or shape[i] for i in range(len(static))]

class GaussianProcess(BaseModel):
    def __init__(self, n_classes, rnn_hidden_dim, mark_emb_dim, layer_hidden_dim, n_layers,
                 n_points, alpha=0, beta=0, lr=0.001, regularization=1e-4, n_samples=10, **kwargs):
        super().__init__(n_classes, rnn_hidden_dim, mark_emb_dim, lr, regularization)
        self.layer_hidden_dim = layer_hidden_dim
        self.n_layers = n_layers
        self.n_points = n_points
        self.alpha = alpha
        self.beta = beta
        self.n_samples = n_samples

        self.log_sigma = tf.get_variable('log-sigma', initializer=tf.constant([1.0]))
        self.epsilon = 0.1

        self.mark_emb = self.mark_embedding(self.x)
        rnn_input = tf.concat([self.mark_emb, tf.expand_dims(self.tx, -1)], -1)
        self.h = self.rnn(rnn_input, self.s)

        self.t_sample, self.y_sample, self.w = self.get_points(self.h)
        self.logits, self.logits_sigma = self.GP(self.t_sample, self.y_sample, self.ty, self.w)

        self.py = tf.nn.softmax(self.logits, -1)
        self.y_hat = tf.argmax(self.logits, -1)

        # sampling loss
        e_k = tf.reduce_sum(tf.exp(self.logits + self.logits_sigma / 2), -1)
        var_k = tf.reduce_sum((tf.exp(self.logits_sigma) - 1) * tf.exp(2 * self.logits + self.logits_sigma), -1)
        mu_k = tf.reduce_sum(tf.one_hot(self.y, n_classes) * self.logits, -1)

        self.loss = -mu_k + tf.log(e_k) - var_k / e_k**2 / 2
        self.loss = self.aggregate(self.loss, self.s)
        self.loss += self.gp_regularization()

        self.optimize = tf.train.AdamOptimizer(lr).minimize(self.loss)
        self.accuracy = tf.identity(self.get_accuracy(), name='accuracy')

    def get_points(self, h):
        shape = get_shape(h)

        t = h
        for _ in range(self.n_layers):
            t = tf.layers.dense(t, self.layer_hidden_dim, tf.nn.relu)
        t = tf.layers.dense(h, self.n_classes * self.n_points, tf.nn.softplus)

        y = h
        for _ in range(self.n_layers):
            y = tf.layers.dense(y, self.layer_hidden_dim, tf.nn.relu)
        y = tf.layers.dense(y, self.n_classes * self.n_points) # [B, S, P * K]

        w = h
        for _ in range(self.n_layers):
            w = tf.layers.dense(w, self.layer_hidden_dim, tf.nn.relu)
        w = tf.layers.dense(y, self.n_classes * self.n_points, tf.sigmoid) # [B, S, P * K]

        t = tf.reshape(t, shape[:-1] + [self.n_classes, self.n_points]) # [B, S, K, P]
        y = tf.reshape(y, shape[:-1] + [self.n_classes, self.n_points]) # [B, S, K, P]
        w = tf.reshape(w, shape[:-1] + [self.n_classes, self.n_points]) # [B, S, K, P]

        return t, y, w

    def GP(self, x, y, _x, w):
        x = tf.expand_dims(x, -1) # [B, S, K, P, 1]
        y = tf.expand_dims(y, -1) # [B, S, K, P, 1]
        # [B, S, K, 1, 1]
        _x = tf.tile(tf.reshape(_x, get_shape(_x) + [1]*3), [1, 1, self.n_classes, 1, 1])

        sigma = self.weight_kernel(w, w) * self.kernel(x, x)
        sigma += self.diagonal_noise(sigma)
        sigma_inverse = tf.matrix_inverse(sigma) # [B, S, K, P, P]

        _w = tf.ones(get_shape(_x)[:2] + [1, 1]) # [B, S, 1, 1]

        k = self.weight_kernel(w, _w) * self.kernel(x, _x) # [B, S, K, P, 1]
        S = tf.matmul(k, sigma_inverse, transpose_a=True) # [B, S, K, 1, P]

        c = self.kernel(_x, _x) # [B, S, K, 1, 1]

        mean = tf.matmul(S, y)
        var = c - tf.matmul(S, k)

        mean = tf.squeeze(mean, [-2, -1]) # [B, S, K]
        var = tf.squeeze(var, [-2, -1]) # [B, S, K]

        return mean, var

    def kernel(self, x1, x2):
        x2 = tf.matrix_transpose(x2)
        return tf.exp(-((x1 - x2) / tf.exp(self.log_sigma))**2)

    def weight_kernel(self, w1, w2):
        return tf.minimum(tf.expand_dims(w1, -1), tf.expand_dims(w2, -2))

    def diagonal_noise(self, A):
        epsilon = tf.ones([get_shape(A)[-1]]) * self.epsilon
        return tf.linalg.tensor_diag(epsilon)

    def gp_regularization(self):
        t = tf.tile(self.t_sample, [self.n_samples, 1, 1, 1])
        y = tf.tile(self.y_sample, [self.n_samples, 1, 1, 1])
        w = tf.tile(self.w, [self.n_samples, 1, 1, 1])
        ty = tf.tile(self.ty, [self.n_samples, 1])
        s = tf.tile(self.s, [self.n_samples])

        _t = tf.random_uniform(get_shape(ty), 0.0, 3.0)

        mean, var = self.GP(t, y, _t, w)

        mean_reg = self.aggregate(tf.reduce_mean(mean**2, -1), s) / self.n_samples
        var_reg = self.aggregate(tf.reduce_mean((1 - var)**2, -1), s) / self.n_samples

        return self.alpha * mean_reg + self.beta * var_reg + self.loss_regularization()
