from src.model import *
from src.utils.utils import *
from src.tokenizer import *
from src.preprocessing_syl import *


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
generate_data(file_training, file_result, file_to_read)
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
model = ModelTransformer(transformer_config, vocab_size, vocab_size)
train_batches = make_batches(train)
val_batches = make_batches(val)
model.train(train_batches, val_batches, 0)  # TODO: remember to change to 20

file_name = 'silvia'
f = open('./poems/' + file_name + '.txt', 'r', encoding='utf-8')
f2 = open('./poems/' + file_name + '_syll.txt', 'w+', encoding='utf-8')
lines = f.readlines()
lines = preprocess_text(lines)
for line in lines:
    line = line.rstrip()
    text_res = model.syllabify(tf.constant(line), tokenizer)
    res = text_res.numpy().decode('utf-8')
    res = res[2:][:-2]
    f2.write(res + '\n')
    print(res)

f.close()
f2.close()

# line = 'nel mezzo del cammin di nostra vita'
# line = 'cantami o diva del pelide achille'
# OK
# line = tf.convert_to_tensor([line])
# OK
# line = tokenizer.tokenize(line).to_tensor()
# encoder_input = line
# text_res = model.syllabify(tf.constant(line), tokenizer)
# print(text_res.numpy().decode('utf-8'))

'''
test_line = make_batches(test)  # line
for (batch, (inp, tar)) in enumerate(test_line):
    encoder_input = inp
    break
'''
# inp, _ = test_line[0]
# encoder_input = inp
'''
start,end = tokenizer.tokenize([''])[0]
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
predicted = text.numpy().decode('utf-8')
print(predicted)
'''

'''
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
    dec_outuput, _ = model.get_transformer().decoder(output, enc_output, False, combined_mask, dec_padding_mask)
    predictions = model.get_transformer().final_layer(dec_outuput)
    predicted_ids = choose_greedy(predictions)

    output = tf.concat([tf.cast(output, dtype=tf.int64), tf.cast(predicted_ids, dtype=tf.int64), ], axis=1)
'''
