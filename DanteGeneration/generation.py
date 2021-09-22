from preprocessing import *
from utility import *
from tokenizer import *
BUFFER_SIZE = 20000
BATCH_SIZE = 64

def tokenize_pairs(X, y):
    X = tokenizer.tokenize(X)
    # Convert from ragged to dense, padding with zeros.
    X = X.to_tensor()

    y = tokenizer.tokenize(y)
    # Convert from ragged to dense, padding with zeros.
    y = y.to_tensor()

    return X, y

def make_batches(ds):
    return (ds
            .cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE)
            )


file_to_read = "divina_syll_good"
file_training = "dante_training_gen"
file_result = "dante_result_training_gen"
file_vocabulary = "dante_vocabulary_gen"


generate_data(file_training,file_result,file_to_read)
train, val, test = load_gen_dataset()

tokenizer = Tokenizer(['S', 'Y', 'T', 'E', 'B' , '[START]', '[END]'],
                      '../outputs_gen/' + file_vocabulary + '.txt')

train_batches = make_batches(train)
val_batches = make_batches(val)

transformer_config = {'num_layers': 6, #4
                      'd_model': 512, #128
                      'num_heads': 8,
                      'dff': 2048, #512
                      'dropout_rate': 0.1}
