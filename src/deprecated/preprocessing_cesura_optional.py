from src.utils.roman_to_int import *
import re
import pickle
import os

# TODO, Mettere EOV EOT, modificarlo per renderlo più simile all'encoding utilizzato da alessandroPaciello, ma ci dovrei mettere poco
# in generateText gran parte del codice è per comprendere dove c'è una sinalefe => importante per la cesura, se non la implementiamo quella parte diventa quasi inutile
# nel codice di pietro poi generano un vocabolario, ma non ho ben capito per quale motivo lo facciano

# add other char if present
punctuation = r'[?!;:.,«»“‟”()-\[\]]'
file_to_read = "divina_tmp"
path_dictionary = '../dictionaries/dantes_dictionary_ass.pkl'


def generateText():
    # open dictionary
    listOfAccents = []
    training, result = readBook()
    dictionary = pickle.load(open(path_dictionary, 'rb'))
    trainingLines = training.split('\n')
    resultLines = result.split('\n')
    map = zip(trainingLines, resultLines)
    # generate list indicating the accents and deal with the sinalefe
    for (el1, el2) in map:
        tonic_accents = []
        # generate a list of words from each sentence
        listOfWords = el1.split()
        length = 0
        for word in listOfWords:
            syllableNumber = dictionary[word][0][0][1]

            length += syllableNumber
        accent = [False for k in range(length)]
        listOfAccents.append(accent)
    map = zip(trainingLines, resultLines)
    for i, (el1, el2) in enumerate(map):
        listOfWords = el1.split()
        j = 0
        for word in listOfWords:
            syllableNumber = dictionary[word][0][0][1]
            accentPosition = dictionary[word][0][0][2][0]
            index = j + syllableNumber + accentPosition - 1
            listOfAccents[i][index] = True
            j += syllableNumber
    map = zip(trainingLines, resultLines)
    # deal with the sinalefe
    for i, (el1, el2) in enumerate(map):
        if len(listOfAccents[i]) != 11:
            syllables = el2.split(f'<SYL>')
            for j, s in enumerate(syllables):
                if bool(re.search(' <SEP> \S', s)):
                    listOfAccents[i].pop(j)
    map = zip(trainingLines, resultLines)
    # dealWithCesura(listOfAccents,map,resultLines)
    data = tf.data.Dataset.from_tensor_slices(trainingLines)
    tokenizer_params = dict(lower_case=False)
    reserved_tokens = ['<SYL>', '<SEP>']


'''
def dealWithCesura(listOfAccents,map,resultLines):
    #delete symbol <SYL> and substitute with <CES>
    #add all possible Cesura
    for i,(el1,el2) in enumerate(map):
        syllables = el2.split(f'<SYL>')
        #one example of cesura, add tag CES
        if listOfAccents[i][5] and syllables[5].endswith('<SEP> '):
            modLine = '<SYL>'.join(syllables[0:6]) + '<CES>' + '<SYL>'.join(syllables[6:])
            print(lineMaschile)
        elif listOfAccents[i][5] and syllables[6].endswith('<SEP> '):
            modLine= '<SYL>'.join(syllables[0:7]) + '<CES>' + '<SYL>'.join(syllables[7:])
    print(map)
    resultLines[i] = modLine
    return resultLines
'''


def readBook():
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
    # delete empty lines, except the first one and the last one
    raw_text = re.sub(r'\n+', '\n', raw_text)
    # delete first empty line
    raw_text = re.sub(r'^\n', '', raw_text)
    # delete last empty line
    raw_text = os.linesep.join([s for s in raw_text.splitlines() if s])
    # generate trainingText and validationText
    trainingText = generate_training_text(raw_text)
    resultText = generate_result_text(raw_text)
    with open('../outputs/danteTraining.txt', 'w+', encoding='utf-8') as file:
        file.writelines(trainingText)
    with open('../outputs/danteResultTraining.txt', 'w+', encoding='utf-8') as file:
        file.writelines(resultText)

    return trainingText, resultText


def generate_training_text(trainingText):
    # delete symbol |
    trainingText = re.sub('\|', '', trainingText)
    return trainingText


def generate_result_text(resultText):
    # add tag sep to indicate separator
    resultText = re.sub(r' +', ' <SEP> ', resultText)

    # add tag syl to indicate syllabification and delete whitespace generated
    resultText = re.sub(r'\|', ' <SYL> ', resultText)
    resultText = re.sub(r'^ ', '', resultText)
    resultText = re.sub(r'\n ', '\n', resultText)
    # delete syl a the beginning of sentences
    resultText = re.sub(r'\n<SYL> ', '\n', resultText)
    resultText = re.sub(r'^<SYL> ', '\n', resultText)
    # some adjustment
    resultText = re.sub(r'^\n', '', resultText)
    resultText = re.sub(r'<SEP>  <SYL>', '<SEP> <SYL>', resultText)
    return resultText


generateText()
