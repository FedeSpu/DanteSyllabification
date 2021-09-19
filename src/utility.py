from sklearn.model_selection import train_test_split

import tensorflow as tf
from src.tokenizer import *
from src.utility import *
import numpy as np


fileTraining = "danteTraining"
fileResult = "danteResultTraining"

train_size = 0.67

def generateDataset():
    with open('../outputs/' + fileTraining + '.txt', 'r+', encoding='utf-8') as file:
        X = file.readlines()
    with open('../outputs/' + fileResult + '.txt', 'r+', encoding='utf-8') as file:
        y = file.readlines()
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=1, train_size = train_size)
    Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, random_state=1, train_size = 0.80)

    train = tf.data.Dataset.from_tensor_slices((Xtrain, ytrain))
    test = tf.data.Dataset.from_tensor_slices((Xtest, ytest))
    val = tf.data.Dataset.from_tensor_slices((Xval, yval))

    return train,val,test

def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
      def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

      def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)






