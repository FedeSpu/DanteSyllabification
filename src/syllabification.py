from src.model import *
from src.utils.utils import *
from src.tokenizer import *
from src.preprocessing_with_tag import *


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


'''
file_name_raw = 'inferno'
file_name_syll = 'inferno_syll'
random_state = 15
'''

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
tokenizer = Tokenizer(['<SEP>', '<SYL>', '<SOV>', '<EOV>', '[START]', '[END]'],
                      '../outputs/' + file_vocabulary + '.txt')

# 2.1) Set hyperparameters
transformer_config = {'num_layers': 4,
                      'd_model': 128,  # 256
                      'num_heads': 8,  # 4
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
dataset = make_batches(train)
model.train(dataset, 10)  # TODO: remember to change to 20

line = 'nel mezzo del cammin di nostra vita'
line = tf.convert_to_tensor([line])
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

start_end = tokenizer.tokenize([''])[0]
start = start_end[0][tf.newaxis]
end = start_end[1][tf.newaxis]

output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
output_array = output_array.write(0, start)
for i in tf.range(10):
    output = tf.transpose(output_array.stack())  # decoder_input
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

    # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
    # enc_output = model.get_transformer().encoder(encoder_input, False, enc_padding_mask)
    tra = model.get_transformer()
    predictions, _ = tra([encoder_input, output], False)

    predictions = predictions[:, -1:, :]

    predicted_id = tf.argmax(predictions, axis=-1)
    output_array = output_array.write(i + 1, predicted_id[0])

    # print(stop_ten)
    if predicted_id == end:
        break
    # print(encoder_input)
    # p, aw = model.get_transformer().call((encoder_input, output), False)

    # for _ in range(10):
    '''
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
    dec_outuput, _ = model.get_transformer().decoder(output, enc_output, False, combined_mask, dec_padding_mask)
    predictions = model.get_transformer().final_layer(dec_outuput)
    predicted_ids = choose_greedy(predictions)

    output = tf.concat([tf.cast(output, dtype=tf.int64), tf.cast(predicted_ids, dtype=tf.int64), ], axis=1)
    '''
# print(output)
stripped_output = tokenizer.detokenize(output)[0]
# list(map(lambda x: x.split('<EOV>')[0], tokenizer.sequences_to_texts(output.numpy())))
# stripped_output = list(map(strip_tokens, stripped_output))
print(stripped_output)
