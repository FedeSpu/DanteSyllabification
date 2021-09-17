from src.preprocessing import *
from sklearn.model_selection import train_test_split
from src.model import *
from src.utils.utils import *
import tensorflow as tf

file_name_raw = 'inferno'
file_name_syll = 'inferno_syll'
random_state = 15

# 1) Manipulate data and prepare DataSet - Tokenization

# Pre-processing of both texts: syllabed and none
verses_raw = ['<GO>', 'Nel mezzo del cammin di nostra vita', 'mi ritrovai per una selva oscura',
              'ché la diritta via era smarrita',
              '<EOF>']  # preprocessing_text(file_name=file_name_raw, file_write=False)
verses_syll = ['<GO>', '|Nel |mez|zo |del |cam|min |di |no|stra |vi|ta', '|mi |ri|tro|vai |per |u|na |sel|va o|scu|ra',
               '|ché |la |di|rit|ta |via |e|ra |smar|ri|ta',
               '<EOF>']  # preprocessing_text(file_name=file_name_syll, file_write=False)

# From raw data to DataSet object, can be used in train a model
# Text needs to be converted to numeric representation
tokenizer = tf.keras.preprocessing.text.Tokenizer(lower=True, char_level=False, filters='')

# Updates internal vocabulary based on a list of texts
tokenizer.fit_on_texts(texts=verses_syll)

# Transforms each text to a sequence of integers
verses_raw_enc = tokenizer.texts_to_sequences(verses_raw)
verses_syll_enc = tokenizer.texts_to_sequences(verses_syll)

# Pad sequences ensure that all sequences in a list have the same length (default adding 0s, post: add 0s after seq)
verses_syll_pad = tf.keras.preprocessing.sequence.pad_sequences(verses_syll_enc, padding='post')
verses_raw_pad = tf.keras.preprocessing.sequence.pad_sequences(verses_raw_enc, padding='post')

# 2) Train the Transformer
# Split data in training and test sets
X_train, X_test, y_train, y_test = train_test_split(verses_raw_pad, verses_syll_pad,
                                                    random_state=random_state, train_size=0.66)

# 2.1) Set hyperparameters
transformer_config = {'num_layers': 4,
                      'd_model': 256,
                      'num_heads': 4,
                      'dff': 1024,
                      'dropout_rate': 0.1}

vocab_size = len(tokenizer.word_index) + 1
model = ModelTransformer(transformer_config, tokenizer, vocab_size, vocab_size)
X_train = tf.dtypes.cast(X_train, dtype=tf.int64)
y_train = tf.dtypes.cast(y_train, dtype=tf.int64)
dataset = make_dataset(X_train, y_train)
model.train(dataset, 2)


def choose_greedy(logits):
    # select the last character from the seq_len dimension
    predicted_ids = tf.argmax(logits[:, -1:, :], axis=-1)
    return predicted_ids


start_symbol = tokenizer.word_index['<go>']
stop_symbol = tokenizer.word_index['<eof>']
encoder_input = verses_raw_pad  # tf.convert_to_tensor(X_test)
# decoder_input = tf.repeat([[start_symbol]], repeats=encoder_input.shape[0], axis=0)
decoder_input = tf.convert_to_tensor([start_symbol])
decoder_input = tf.expand_dims(decoder_input, 0)

output = decoder_input
enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
# enc_output = model.get_transformer().encoder(encoder_input, False, enc_padding_mask)
p, aw = model.get_transformer().call((encoder_input, output), False)

for _ in range(100):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
    dec_outuput, _ = model.get_transformer().decoder(output, enc_output, False, combined_mask, dec_padding_mask)
    predictions = model.get_transformer().final_layer(dec_outuput)
    predicted_ids = choose_greedy(predictions)

    output = tf.concat([tf.cast(output, dtype=tf.int64), tf.cast(predicted_ids, dtype=tf.int64), ], axis=1)

print(output)