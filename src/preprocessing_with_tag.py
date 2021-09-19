from src.utils.roman_to_int import *
import re
import pickle
import os
import tensorflow as tf
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab


file_to_read = "divina_syll_good"
fileTraining = "danteTraining"
fileResult = "danteResultTraining"
fileVocabulary = "danteVocabulary"
punctuation = r'[?!;:.,«»“‟”()-\[\]]'

def generateData():
    data = readData()
    data, trainingData = generateTrainingData(data)
    result = generateResult(data)
    with open('../outputs/'+ fileTraining +'.txt', 'w+', encoding='utf-8') as file:
        file.writelines(trainingData)
    with open('../outputs/'+ fileResult +'.txt', 'w+', encoding='utf-8') as file:
        file.writelines(result)
    #nel codice di pietro viene applicato questo ma i risultati sono prettamente simili, la computazione un po' più veloce ma non so il motivo per cui lo usino
    '''
    text_no_tag = re.sub(rf'<SYL>', ' ', result)
    text_no_tag = re.sub(rf'<SEP>', ' ', text_no_tag)
    text_no_tag = re.sub(rf'<SOV>', ' ', text_no_tag)
    text_no_tag = re.sub(rf'<EOV>', ' ', text_no_tag)
    text_no_tag = re.sub(r' +', f' ', text_no_tag)
    # remove spaces at the beginning of each line
    text_no_tag = re.sub(r'^ ', '', text_no_tag)
    text_no_tag = re.sub(r'\n ', '\n', text_no_tag)
    '''
    generateVocabulary(trainingData)


def generateTrainingData(data):
    # delete empty lines, except the first one and the last one
    data = re.sub(r'\n+', '\n', data)
    # delete first empty line
    data = re.sub(r'^\n', '', data)
    # delete last empty line
    data = data.rstrip('\n')
    #delte whitespace
    data = re.sub(r' *\n','\n',data)
    data = re.sub(r' *$', '', data)
    #delete | indicating syllabification
    trainingData = re.sub(r'\|', '', data)
    return data,trainingData

def generateResult(data):
    # add tag sep to indicate separator
    resultText = re.sub(r' +', ' <SEP> ', data)
    # add tag syl to indicate syllabification and delete whitespace generated
    resultText = re.sub(r'\|', ' <SYL> ', resultText)
    #adjustment
    resultText = re.sub(r'<SEP>  <SYL>', '<SEP> <SYL>', resultText)
    resultText = re.sub(r'\n <SYL>', '\n<SYL>', resultText)
    #add SOV as start of verse
    resultText = re.sub(r'\n<SYL>','\n<SOV> <SYL>',resultText)
    #add EOV as end of verse
    resultText = re.sub(r'<SEP> \n','<SEP> <EOV>\n',resultText)
    # add SOV as start of the first verse
    resultText = re.sub(r'^ <SYL>', '\n<SOV> <SYL>', resultText)
    # add EOV as end of last verse
    resultText = re.sub(r'<SEP> $', '<SEP> <EOV>\n', resultText)
    # delete first empty line
    resultText = re.sub(r'^\n', '', resultText)
    return resultText



def readData():
    with open('../text/' + file_to_read + '.txt', 'r+', encoding='utf-8') as file:
        raw_text = file.read()
    # convert into lower case
    raw_text = raw_text.lower()
    # remove sentences such as canto V
    raw_text = re.sub(r'.* • canto .*', '', raw_text)
    # remove enumeration
    raw_text = re.sub(r'\n *\d* ', '\n', raw_text)
    # replace auxiliary characters
    raw_text = re.sub(r'[’‘\']', '’', raw_text)
    # remove punctuation
    raw_text = re.sub(punctuation, '', raw_text)
    # delete first empty line
    raw_text = re.sub(r'^\n\n', '', raw_text)
    return raw_text


def generateVocabulary(trainingData):
    train_pt = tf.data.Dataset.from_tensor_slices(trainingData.split('\n'))
    print(trainingData.split('\n'))
    bert_tokenizer_params = dict(lower_case=True)
    reserved_tokens = ['<SEP>','<SYL>' ,'<SOV>','<EOV>','[START]','[END]']
    bert_vocab_args = dict(
        # The target vocabulary size
        vocab_size=200, #capire perchè è 200 e non 1000/2000
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
    )
    with open('../outputs/'+ fileVocabulary +'.txt', 'w', encoding='utf-8') as f:
        for token in pt_vocab:
            print(token, file=f)


generateData()
