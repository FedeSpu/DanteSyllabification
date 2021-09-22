from sklearn.model_selection import train_test_split
import tensorflow as tf

# file_training = "danteTraining"
# file_result = "danteResultTraining"

train_size = 0.98
train_size_val = 0.99
random_state = 1


def generate_dataset(file_training, file_result):
    #OK
    with open('../outputs/' + file_training + '.txt', 'r+', encoding='utf-8') as file:
        X = file.readlines()
    with open('../outputs/' + file_result + '.txt', 'r+', encoding='utf-8') as file:
        y = file.readlines()
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=random_state, train_size=train_size)
    # TODO: check validation
    Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, random_state=random_state, train_size=train_size_val)

    train = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain))
    test = tf.data.Dataset.from_tensor_slices((Xtest, ytest))
    val = tf.data.Dataset.from_tensor_slices((Xval, yval))
    return train, val, test

'''
def tokenize_pairs(X, y):
    X = tokenizer.tokenize(X)
    # Convert from ragged to dense, padding with zeros.
    X = X.to_tensor()

    y = tokenizer.tokenize(y)
    # Convert from ragged to dense, padding with zeros.
    y = y.to_tensor()

    return X, y


def make_batches(ds, BUFFER_SIZE, BATCH_SIZE):
    return (
        ds
            .cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))
'''
