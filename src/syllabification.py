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
verses_raw = preprocessing_text(file_name=file_name_raw, file_write=False)
verses_syll = preprocessing_text(file_name=file_name_syll, file_write=False)

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
model.train(dataset, 40)
