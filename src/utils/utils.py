# TODO: create batches in out project
import tensorflow as tf
'''
BUFFER_SIZE = 20000
BATCH_SIZE = 64


def tokenize_pairs(pt, en):
    pt = tokenizers.pt.tokenize(pt)
    # Convert from ragged to dense, padding with zeros.
    pt = pt.to_tensor()

    en = tokenizers.en.tokenize(en)
    # Convert from ragged to dense, padding with zeros.
    en = en.to_tensor()
    return pt, en


def make_batches(ds):
    return (
        ds
            .cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))


train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)
'''


def make_dataset(*sequences, batch_size=64):
    buffer_size = len(sequences[0])

    dataset = tf.data.Dataset.from_tensor_slices(tuple(sequences)).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset
