from src.utils.roman_to_int import *
import re
import pickle
import os

file_to_read = "divina_tmp"
fileTraining = "danteTraining"
fileResult = "danteResultTraining"
punctuation = r'[?!;:.,«»“‟”()-\[\]]'

def generateData():
    data = readData()
    trainingData = re.sub(r'\|', '', data)
    result = generateResult(data)
    with open('../outputs/'+ fileTraining +'.txt', 'w+', encoding='utf-8') as file:
        file.writelines(trainingData)
    with open('../outputs/'+ fileResult +'.txt', 'w+', encoding='utf-8') as file:
        file.writelines(result)

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


generateData()
