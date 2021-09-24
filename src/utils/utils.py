from sklearn.model_selection import train_test_split
import tensorflow as tf
import re
from matplotlib import pyplot as plt
import numpy as np

# file_training = "danteTraining"
# file_result = "danteResultTraining"

train_size = 0.98
train_size_val = 0.99
random_state = 1
'''
train_size = 0.95
train_size_val = 0.8
random_state = 42
'''

def generate_dataset(file_training, file_result):
    with open('../outputs/' + file_training + '.txt', 'r+', encoding='utf-8') as file:
        X = file.readlines()
    with open('../outputs/' + file_result + '.txt', 'r+', encoding='utf-8') as file:
        y = file.readlines()
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=random_state, train_size=train_size)
    # TODO: check validation
    Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, random_state=random_state, train_size=train_size_val)

    train = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain))
    print("train")
    print(len(list(train)))
    test = tf.data.Dataset.from_tensor_slices((Xtest, ytest))
    print("test")
    print(len(list(test)))
    val = tf.data.Dataset.from_tensor_slices((Xval, yval))
    print("val")
    print(len(list(val)))

    return train, val, test


def plot_accuracy(train_losses, train_accuracies, val_losses, val_accuracies):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.set_title('Accuracy')
    ax1.set(xlabel='Epoch')
    ax1.plot(train_accuracies, label='Train')
    ax1.plot(val_accuracies, label='Validation')
    start, end = ax1.get_xlim()
    start = 0
    end = 20
    ax1.set_xticks(np.arange(start, end, 3))

    ax2.set_title('Loss')
    ax2.set(xlabel='Epoch')
    ax2.plot(train_losses, label='Train')
    ax2.plot(val_losses, label='Validation')
    start, end = ax2.get_xlim()
    start = 0
    end = 20
    ax2.set_xticks(np.arange(start, end, 3))
    plt.show()

def load_gen_dataset():
    with open('../outputs_gen/dante_training_gen.txt', 'r+', encoding='utf-8') as file:
        X = file.readlines()
    with open('../outputs_gen/dante_result_training_gen.txt', 'r+', encoding='utf-8') as file:
        y = file.readlines()
    X=''.join(X)
    y = ''.join(y)
    X = re.sub(r'\n',' ',X)
    y = re.sub(r'\n',' ', y)
    X = X.split('T')
    y = y.split('T')
    #TODO: controllare meglio i settaggi
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=random_state, train_size=0.75)
    Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, random_state=random_state, train_size=0.80)
    train = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain))
    test = tf.data.Dataset.from_tensor_slices((Xtest, ytest))
    val = tf.data.Dataset.from_tensor_slices((Xval, yval))
    return train,test,val
