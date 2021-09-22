from src.utils.roman_to_int import *
import re
import pickle
import os
import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab

file_vocabulary = "dante_vocabulary"
punctuation = r'[?!;:.,«»"“‟”()\-—\[\]]'


# Pre-processing text and produce output file
def generate_data(file_training, file_result, file_to_read):
    data = read_data(file_to_read)
    data, training_data = generate_training_data(data)
    result = generate_result(data)
    with open('../outputs/' + file_training + '.txt', 'w+', encoding='utf-8') as file:
        file.writelines(training_data)
    with open('../outputs/' + file_result + '.txt', 'w+', encoding='utf-8') as file:
        file.writelines(result)
    # TODO: nel codice di pietro viene applicato questo ma i risultati sono prettamente simili, la computazione un po' più veloce ma non so il motivo per cui lo usino
    text_no_tag = re.sub(rf'Y', ' ', result)
    text_no_tag = re.sub(rf'S', ' ', text_no_tag)
    text_no_tag = re.sub(rf'T', ' ', text_no_tag)
    text_no_tag = re.sub(rf'E', ' ', text_no_tag)
    text_no_tag = re.sub(r' +', f' ', text_no_tag)
    # remove spaces at the beginning of each line
    text_no_tag = re.sub(r'^ ', '', text_no_tag)
    text_no_tag = re.sub(r'\n ', '\n', text_no_tag)
    generate_vocabulary(text_no_tag)


# Generate text not syllabied
def generate_training_data(data):
    # delete empty lines, except the first one and the last one
    data = re.sub(r'\n+', '\n', data)
    # delete first empty line
    data = re.sub(r'^\n', '', data)
    # delete last empty line
    data = data.rstrip('\n')
    # delete whitespace
    data = re.sub(r'\n *', '\n', data)
    data = re.sub(r' *\n', '\n', data)
    data = re.sub(r' *$', '', data)
    # delete | indicating syllabification
    training_data = re.sub(r'\|', '', data)
    return data, training_data


# Generate text syllabied with tag
def generate_result(data):
    # add tag sep to indicate separator
    result_text = re.sub(r' +', ' S ', data)
    # add tag syl to indicate syllabification and delete whitespace generated
    result_text = re.sub(r'\|', ' Y ', result_text)
    # adjustment
    result_text = re.sub(r'S  Y', 'S Y', result_text)
    result_text = re.sub(r'\n Y', '\nY', result_text)
    # add SOV as start of verse
    result_text = re.sub(r'\nY', '\nT Y', result_text)
    # add EOV as end of verse
    result_text = re.sub(r'\n', ' S E\n', result_text)
    # add SOV as start of the first verse
    result_text = re.sub(r'^ Y', '\nT Y', result_text)
    # add EOV as end of last verse
    result_text = re.sub(r'$', ' S E\n', result_text)
    # delete first empty line
    result_text = re.sub(r'^\n', '', result_text)
    return result_text


# Read file syllabied, transform for handling
def read_data(file_to_read):
    with open('../text/' + file_to_read + '.txt', 'r+', encoding='utf-8') as file:
        raw_text = file.read()
    # convert into lower case
    raw_text = raw_text.lower()
    # remove sentences such as canto V
    raw_text = re.sub(r'.* • canto .*', '', raw_text)
    # remove punctuation
    raw_text = re.sub(punctuation, '', raw_text)
    # remove enumeration
    raw_text = re.sub(r'\n *\d* ', '\n', raw_text)
    # replace auxiliary characters
    raw_text = re.sub(r'[’‘\']', '’', raw_text)
    # delete first empty line
    raw_text = re.sub(r'^\n\n', '', raw_text)
    return raw_text


# Generate vocabulary for tokenizer
def generate_vocabulary(training_data):
    train_pt = tf.data.Dataset.from_tensor_slices(training_data.split('\n'))
    bert_tokenizer_params = dict(lower_case=True)
    reserved_tokens = ['S', 'Y', 'T', 'E', '[START]', '[END]']
    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size=200,  # TODO: capire perchè è 200 e non 1000/2000
        # Reserved tokens that must be included in the vocabulary
        reserved_tokens=reserved_tokens,
        # Arguments for `text.BertTokenizer`
        bert_tokenizer_params=bert_tokenizer_params,
        # Arguments for `wordpiece_vocab.wordpiece_tokenizer_learner_lib.learn`
        learn_params={},
    )

    pt_vocab = bert_vocab.bert_vocab_from_dataset(
        train_pt.batch(1000).prefetch(2),
        **bert_vocab_args
    )  # TODO: 1000 or 200?
    with open('../outputs/' + file_vocabulary + '.txt', 'w', encoding='utf-8') as f:
        for token in pt_vocab:
            print(token, file=f)


file_training = 'dante_training'
file_result = 'dante_result_training'
file_to_read = 'divina_syll_good'


generate_data(file_training, file_result, file_to_read)