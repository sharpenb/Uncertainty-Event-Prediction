import os
import logging
import tensorflow as tf
import numpy as np

from data import get_dataset

from models.gp import GaussianProcess
from models.dirichlet import Dirichlet
from models.dpp import DirichletPointProcess

models = {
    'gp': GaussianProcess,
    'dirichlet': Dirichlet,
    'dpp': DirichletPointProcess
}

## General config
model_name = 'dirichlet'      # ['gp', 'dirichlet', 'dpp']
dataset_name = 'random_graph' # ['mooc', 'random_graph', 'smart-home-A', 'stack_overflow']
max_epochs = 1000             # Maximum number of epochs
patience = 10                 # After how many iterations to stop the training
batch_size = 32               # How many sequences in each batch during training
rnn_hidden_dim = 64           # Size of RNN hidden state
mark_emb_dim = 64             # Size of input mark embedding vector
layer_hidden_dim = 64         # Size of a hidden layer that generates pseude points from RNN hidden state
n_layers = 2                  # Number of layers that generate points (for GP)
n_points = 20                 # Number of points to generate
n_samples = 10                # Number of samples to use in Monte Carlo estimations (if used)
alpha = 1e-3                  # Alpha regularization param (eq. 7)
beta = 1e-3                   # Beta regularization param (eq. 7)
lr = 1e-3                     # Learning rate of Adam optimizer
regularization = 1e-3         # L2 regularization

## Datasets
train_dataset, val_dataset, test_dataset, num_classes = get_dataset(dataset_name)

train_dataset = train_dataset.batch(batch_size).shuffle(buffer_size=10000000)
val_dataset = val_dataset.batch(1)
test_dataset = test_dataset.batch(1)

train_iterator = train_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()
test_iterator = test_dataset.make_initializable_iterator()

train_next_element = train_iterator.get_next()
val_next_element = val_iterator.get_next()
test_next_element = test_iterator.get_next()

## Model
m = models[model_name](num_classes,
                       rnn_hidden_dim=rnn_hidden_dim,
                       mark_emb_dim=mark_emb_dim,
                       layer_hidden_dim=layer_hidden_dim,
                       n_layers=n_layers,
                       n_points=n_points,
                       alpha=alpha,
                       beta=beta,
                       n_samples=n_samples,
                       lr=lr,
                       regularization=regularization)

## Training
def training_loop(sess, next_element, optimize=True):
    losses, accuracies = [], []
    while True:
        try:
            tx, ty, x, y, s = sess.run(next_element)
            feed = { m.tx: tx, m.ty: ty, m.x: x, m.y: y, m.s: s }
            if optimize:
                _, loss, acc = sess.run([m.optimize, m.loss, m.accuracy], feed)
            else:
                loss, acc = sess.run([m.loss, m.accuracy], feed)
            losses.append(loss)
            accuracies.append(acc)
        except tf.errors.OutOfRangeError:
            if optimize:
                return np.mean(losses), np.mean(accuracies)
            else:
                return np.mean(losses), np.mean(accuracies)

impatient = 0
best_loss = np.inf
training_val_losses = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(max_epochs):
        sess.run(train_iterator.initializer)
        sess.run(val_iterator.initializer)

        # Train
        train_loss, train_acc = training_loop(sess, train_next_element)
        # Validation
        val_loss, val_acc = training_loop(sess, val_next_element, optimize=False)
        training_val_losses.append(val_loss)
        print(f'Train loss {train_loss:.3f} acc {train_acc:.3f} -- Val loss {val_loss:.3f} acc {val_acc:.3f}')

        # Check early stopping criterion
        if val_loss > best_loss:
            impatient += 1
        else:
            impatient = 0
            best_loss = val_loss

        if impatient > patience:
            print(f'Early stopping at {epoch} epoch')
            break

    # Test
    sess.run(test_iterator.initializer)
    test_loss, test_acc = training_loop(sess, test_next_element, optimize=False)
    print(f'Test loss {test_loss:.3f} acc {test_acc:.3f}')
