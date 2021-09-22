import re
from sklearn.model_selection import train_test_split
import tensorflow as tf

random_state=1

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