import tensorflow as tf
import numpy as np
import tensorflow.contrib.distributions as tfp
from .base import BaseModel

def get_shape(x):
    static = x.get_shape().as_list()
    shape = tf.shape(x)
    return [static[i] or shape[i] for i in range(len(static))]

class DirichletPointProcess(BaseModel):
    def __init__(self, n_classes, rnn_hidden_dim, mark_emb_dim, layer_hidden_dim, n_layers,
                 n_points, lr=0.001, regularization=1e-4, n_samples=10, **kwargs):
        super().__init__(n_classes, rnn_hidden_dim, mark_emb_dim, lr, regularization)
        self.layer_hidden_dim = layer_hidden_dim
        self.n_gaussians = n_points
        self.n_samples = n_samples

        mark_emb = self.mark_embedding(self.x)
        rnn_input = tf.concat([mark_emb, tf.expand_dims(self.tx, -1)], -1)
        self.h = self.rnn(rnn_input, self.s)

        self.logits, self.pi, self.means, self.vars = self.Dirichlet(self.h, self.ty)
        self.alpha_t = tf.exp(self.logits)
        self.alpha0 = tf.reduce_sum(tf.exp(self.logits), axis=-1)

        self.py = tf.nn.softmax(self.logits)
        self.y_hat = tf.argmax(self.logits, -1)

        self.loss = self.get_loss()
        self.loss += self.loss_regularization()
        self.optimize = self.optimize_step(self.loss)
        self.accuracy = self.get_accuracy()

    def Dirichlet(self, h, ty, scope='dirichlet', reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            # Compute means and variances of gaussians
            means = tf.layers.dense(h, self.n_gaussians * self.n_classes, activation=tf.sigmoid, kernel_initializer=tf.random_uniform_initializer, name='layer-means')
            means = tf.reshape(means, [-1, tf.shape(means)[1], self.n_classes, self.n_gaussians])
            vars = tf.layers.dense(h, self.n_gaussians * self.n_classes, activation=tf.nn.softplus, kernel_initializer=tf.random_uniform_initializer, name='layer-vars')
            vars = tf.reshape(vars, [-1, tf.shape(vars)[1], self.n_classes, self.n_gaussians])

            # Compute gaussians probabilities
            ty = tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(ty, -1), [1, 1, self.n_classes]), -1), [1, 1, 1, self.n_gaussians])
            gaussians = tf.distributions.Normal(loc=means, scale=vars).prob(ty)

            # Compute log of alphas by combining gaussians
            unstacked = tf.keras.layers.Lambda(lambda x: tf.unstack(x, axis=-2))(gaussians)
            dense_outputs = []
            dense_pis = []
            W_norm = {}
            for k, x in enumerate(unstacked):
                W_norm[k] = tf.layers.dense(h, self.n_gaussians)
                W_norm[k] = tf.expand_dims(W_norm[k], -2)
                x = tf.expand_dims(x, -1)
                dense_outputs.append(tf.squeeze(tf.matmul(W_norm[k], x), [-1]))
                dense_pis.append(W_norm[k])
            log_alpha_t = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-1))(dense_outputs)
            pi = tf.keras.layers.Lambda(lambda x: tf.concat(x, axis=-2))(dense_pis)
            return log_alpha_t, pi, means, vars

    def get_loss(self):
        # First term
        dirichlet_expectation = tf.digamma(self.alpha_t) - tf.tile(tf.expand_dims(tf.digamma(self.alpha0), -1), [1, 1, self.n_classes])
        y = tf.one_hot(self.y, self.n_classes)
        self.loss_ent_unc = -self.aggregate(tf.reduce_sum(dirichlet_expectation * y, -1), self.s)

        # Second term
        self.loss_pp = self.aggregate(-tf.log(self.alpha0 + 1e-8), self.s)

        x = tf.tile(self.x, [self.n_samples, 1])
        tx = tf.tile(tf.expand_dims(self.tx, -1), [self.n_samples, 1, 1])
        ty = tf.tile(self.ty, [self.n_samples, 1])
        s = tf.tile(self.s, [self.n_samples])

        t_sample = ty * tf.random_uniform(tf.shape(ty), 0, 1)

        rnn_input = tf.concat([self.mark_embedding(x), tx], -1)
        h = self.rnn(rnn_input, s, reuse=True)

        log_alpha_t_sample, _, _, _ = self.Dirichlet(h, t_sample)
        prior = 0
        alpha_t_sample = tf.exp(log_alpha_t_sample - prior)

        alpha_0_sample = ty * tf.reduce_sum(alpha_t_sample, axis=-1)
        loss_pp = tf.reshape(alpha_0_sample, [self.n_samples] + get_shape(self.x))  # [10, B, S]
        loss_pp = tf.reduce_mean(loss_pp, 0)  # [10, B, S] -> [B, S]
        self.loss_pp += tf.reduce_mean(loss_pp)

        loss = self.loss_ent_unc + self.loss_pp

        return loss
