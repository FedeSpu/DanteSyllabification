

train_size = 0.67
from sklearn.model_selection import train_test_split
import tensorflow as tf

fileTraining = "danteTraining"
fileResult = "danteResultTraining"

def generateDataset():
    with open('../outputs/' + fileTraining + '.txt', 'r+', encoding='utf-8') as file:
        X = file.readlines()
    with open('../outputs/' + fileResult + '.txt', 'r+', encoding='utf-8') as file:
        y = file.readlines()
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1, train_size = train_size)
    Xtrain = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain))
    ytest = tf.data.Dataset.from_tensor_slices((Xtest, ytest))
    dataset = {'train': Xtrain, 'test': ytest}
    return dataset

generateDataset()