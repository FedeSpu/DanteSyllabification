from src.model import *
from src.tokenizer_gen import *
from src.preprocessing_gen import *
from src.utils.utils import *

# tf.test.gpu_device_name()
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)

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

# generate_data(file_training, file_result, file_to_read)
train, val, test = load_gen_dataset()

tokenizer = TokenizerGen(['S', 'Y', 'T', 'E', 'B', '[START]', '[END]'],
                         '../outputs_gen/' + file_vocabulary + '.txt')

train_batches = make_batches(train)
val_batches = make_batches(val)

transformer_config = {'num_layers': 4,  # 4
                      'd_model': 256,  # 218
                      'num_heads': 4,
                      'dff': 512,  # 512
                      'dropout_rate': 0.1}

vocab_size = tokenizer.get_vocab_size().numpy() + 1
model = ModelTransformer(transformer_config, vocab_size, vocab_size)
train_batches = make_batches(train)  # dataset = make_batches(train) (Codice Fede)
val_batches = make_batches(val)  # dataset = make_batches(val)   (Codice Fede)
model.train(train_batches, val_batches, 0)  # TODO: remember to change to 20

sentence = 'ove udirai le disperate strida vedrai li antichi spiriti dolenti ch’ a la seconda morte ciascun grida'
print(model.generate(tf.constant(sentence), tokenizer))
