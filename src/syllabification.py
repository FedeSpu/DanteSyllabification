from src.model import *
from src.utils.utils import *
from src.tokenizer import *
from src.preprocessing_with_tag import *


# OK
def tokenize_pairs(X, y):
    X = tokenizer.tokenize(X)
    # Convert from ragged to dense, padding with zeros.
    X = X.to_tensor()

    y = tokenizer.tokenize(y)
    # Convert from ragged to dense, padding with zeros.
    y = y.to_tensor()

    return X, y


# OK
def make_batches(ds):
    return (
        ds
            .cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))


# OK
file_training = 'dante_training'
file_result = 'dante_result_training'
file_to_read = 'divina_syll_good'
file_vocabulary = 'dante_vocabulary'

# OK
BUFFER_SIZE = 20000
BATCH_SIZE = 64


def make_dataset(*sequences, batch_size=64):
    buffer_size = len(sequences[0])

    dataset = tf.data.Dataset.from_tensor_slices(tuple(sequences)).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset


# 1) Pre-processing data
# Pre-processing
# generate_data(file_training, file_result, file_to_read)
# Generate train, validation and test data
train, val, test = generate_dataset(file_training, file_result)
# Tokenization
tokenizer = Tokenizer(['S', 'Y', 'T', 'E', '[START]', '[END]'],
                      '../outputs/' + file_vocabulary + '.txt')

# 2.1) Set hyperparameters
transformer_config = {'num_layers': 4,
                      'd_model': 256,  # 128
                      'num_heads': 4,  # 8
                      'dff': 512,  # 1024
                      'dropout_rate': 0.1}

# vocab_size = len(tokenizer.word_index) + 1
vocab_size = tokenizer.get_vocab_size().numpy() + 1
'''
model = ModelTransformer(transformer_config, tokenizer, vocab_size, vocab_size)
X_train = tf.dtypes.cast(X_train, dtype=tf.int64)
y_train = tf.dtypes.cast(y_train, dtype=tf.int64)
dataset = make_dataset(X_train, y_train)
'''
model = ModelTransformer(transformer_config, tokenizer, vocab_size, vocab_size)
train_batches = make_batches(train)  # dataset = make_batches(train) (Codice Fede)
val_batches = make_batches(val)  # dataset = make_batches(val)   (Codice Fede)
model.train(train_batches, val_batches, 0)  # TODO: remember to change to 20

# losses = get_loss_funcs()
# plot_accuracy(losses[0], losses[1], losses[2], losses[3])

line = "cantami o diva del pelide achille"
# OK
line = tf.convert_to_tensor([line])
# OK
line = tokenizer.tokenize(line).to_tensor()
encoder_input = line
'''
test_line = make_batches(test)  # line
for (batch, (inp, tar)) in enumerate(test_line):
    encoder_input = inp
    break
'''
# inp, _ = test_line[0]
# encoder_input = inp
'''
start, end = tokenizer.tokenize([''])[0]
output = tf.convert_to_tensor([start])
'''
# start = tf.convert_to_tensor(2, dtype=tf.int64)
# end = tf.convert_to_tensor(3, dtype=tf.int64)
start, end = tokenizer.tokenize([''])[0]
output = tf.convert_to_tensor([start])
output = tf.expand_dims(output, 0)
tra = model.get_transformer()

for i in range(100):
    enc_padding_mask, combined_mask, dec_padding_mask = tra.create_masks(encoder_input, output)
    predictions, attention_weights = tra.call((encoder_input, output), False)

    predictions = predictions[:, -1:, :]

    predicted_id = tf.argmax(predictions, axis=-1)
    output = tf.concat([output, predicted_id], axis=-1)

    # print(stop_ten)
    if predicted_id == end:
        break

text = tokenizer.detokenize(output)[0]  # shape: ()

'''
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
    dec_outuput, _ = model.get_transformer().decoder(output, enc_output, False, combined_mask, dec_padding_mask)
    predictions = model.get_transformer().final_layer(dec_outuput)
    predicted_ids = choose_greedy(predictions)

    output = tf.concat([tf.cast(output, dtype=tf.int64), tf.cast(predicted_ids, dtype=tf.int64), ], axis=1)
'''
predicted = text.numpy().decode('utf-8')
print(predicted)
