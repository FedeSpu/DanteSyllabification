from src.model import *
from src.utils.utils import *
from src.tokenizer import *
from src.preprocessing_syl import *


def tokenize_pairs(X, y):
    X = tokenizer.tokenize(X)
    # Convert from ragged to dense, padding with zeros.
    X = X.to_tensor()

    y = tokenizer.tokenize(y)
    # Convert from ragged to dense, padding with zeros.
    y = y.to_tensor()

    return X, y


def make_batches(ds):
    return (
        ds
            .cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))


file_training = 'dante_training'
file_result = 'dante_result_training'
file_to_read = 'divina_syll_good'
file_vocabulary = 'dante_vocabulary'


BUFFER_SIZE = 20000
BATCH_SIZE = 64


def make_dataset(*sequences, batch_size=64):
    buffer_size = len(sequences[0])

    dataset = tf.data.Dataset.from_tensor_slices(tuple(sequences)).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


# 1) Pre-processing data
# Pre-processing
generate_data(file_training, file_result, file_to_read)
# Generate train, validation and test data
train, val, test = generate_dataset(file_training, file_result)
# Tokenization
tokenizer = Tokenizer(['S', 'Y', 'T', 'E', '[START]', '[END]'],
                      '../outputs/' + file_vocabulary + '.txt')

# 2.1) Set hyperparameters
transformer_config = {'num_layers': 4,
                      'd_model': 128,  # 256
                      'num_heads': 8,  # 4
                      'dff': 512,  # 1024
                      'dropout_rate': 0.1}

# vocab_size = len(tokenizer.word_index) + 1
vocab_size = tokenizer.get_vocab_size().numpy()+1

model = ModelTransformer(transformer_config, vocab_size, vocab_size)
train_batches = make_batches(train)
val_batches = make_batches(val)
model.train(train_batches,val_batches, 2) #

line = 'nel mezzo del cammin di nostra vita'
text=model.syllabify(tf.constant(line),tokenizer)
print(text)
