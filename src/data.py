import numpy as np
import tensorflow as tf
from pathlib import Path

DATA_DIR = Path(__file__).parents[1] / 'data'


def load_dataset(name):
    """Load dataset."""
    if not name.endswith('.npz'):
        name += '.npz'
    loader = dict(np.load(DATA_DIR / name, allow_pickle=True))
    deltas = loader['deltas']
    marks = loader['marks']
    num_classes = len(set([x for s in marks for x in s]))
    return deltas, marks, num_classes

def list_datasets():
    check = lambda x: x.is_file() and x.suffix == '.npz'
    file_list = [x.stem for x in (DATA_DIR).iterdir() if check(x)]
    return file_list

def split_on_sequence(data):
    l = data.shape[1]
    return data[:,:int(0.6*l)], data[:,int(0.6*l):int(0.8*l)], data[:,int(0.8*l):]

def split_on_data(data):
    l = data.shape[0]
    return data[:int(0.6*l)], data[int(0.6*l):int(0.8*l)], data[int(0.8*l):]

def break_down_sequences(seq, max_len):
    new_seq = []
    for subseq in seq:
        new_seq += [subseq[i:i + max_len] for i in range(0, len(subseq), max_len)]
    return new_seq

def pad_sequence(seq, max_len):
    for i in range(len(seq)):
        seq[i] = np.concatenate([seq[i], [0.] * (max_len - len(seq[i]))])
    return np.stack(seq)

def create_dataset(marks, deltas, lengths):
    in_times = tf.convert_to_tensor(deltas[:,:-1], np.float32)
    out_times = tf.convert_to_tensor(deltas[:,1:], np.float32)
    in_marks = tf.convert_to_tensor(marks[:,:-1], np.int32)
    out_marks = tf.convert_to_tensor(marks[:,1:], np.int32)
    lengths = tf.convert_to_tensor(lengths, np.int32)

    assert in_times.shape == out_times.shape == in_marks.shape == out_marks.shape
    assert in_times.shape[0] == lengths.shape[0]

    dataset = tf.data.Dataset.from_tensor_slices((in_times, out_times, in_marks, out_marks, lengths))
    return dataset

def get_dataset(name, max_training_seq_len=64, deltas=None, marks=None, num_classes=None):
    # Add 1 because in_time is 0:max_training_seq_len-1
    max_training_seq_len += 1

    if name is not None and marks is None and deltas is None:
        deltas, marks, num_classes = load_dataset(name)

    if marks.shape[0] == 1:
        # If only one long sequence, split on it
        train_marks, val_marks, test_marks = split_on_sequence(marks)
        train_deltas, val_deltas, test_deltas = split_on_sequence(deltas)
    else:
        # If multiple sequences, split such that each set has whole sequences
        train_marks, val_marks, test_marks = split_on_data(marks)
        train_deltas, val_deltas, test_deltas = split_on_data(deltas)

    # Break down long training sequences to enable batch training
    train_marks = break_down_sequences(train_marks, max_training_seq_len)
    train_deltas = break_down_sequences(train_deltas, max_training_seq_len)

    # Lengths of all the sequences
    train_lengths = [len(x) - 1 for x in train_marks]
    val_lengths = [len(x) - 1 for x in val_marks]
    test_lengths = [len(x) - 1 for x in test_marks]

    # Add 0 padding so all sequences have same length
    train_marks = pad_sequence(train_marks, max_training_seq_len)
    train_deltas = pad_sequence(train_deltas, max_training_seq_len)

    val_marks = pad_sequence(val_marks, max(val_lengths) + 1)
    val_deltas = pad_sequence(val_deltas, max(val_lengths) + 1)

    test_marks = pad_sequence(test_marks, max(test_lengths) + 1)
    test_deltas = pad_sequence(test_deltas, max(test_lengths) + 1)

    # Create tf.data.Dataset
    train_dataset = create_dataset(train_marks, train_deltas, train_lengths)
    val_dataset = create_dataset(val_marks, val_deltas, val_lengths)
    test_dataset = create_dataset(test_marks, test_deltas, test_lengths)

    return train_dataset, val_dataset, test_dataset, num_classes
