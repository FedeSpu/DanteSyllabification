from src.tokenizer import *
from src.utility import *


import matplotlib.pyplot as plt


fileTraining = "danteTraining"
fileResult = "danteResultTraining"

BUFFER_SIZE = 20000
BATCH_SIZE = 64

#can't move these two functions in util.py, gives an error
def tokenize_pairs(X, y):
    X = tokenizer.tokenize(X)
    # Convert from ragged to dense, padding with zeros.
    X = X.to_tensor()

    y = tokenizer.tokenize(y)
    # Convert from ragged to dense, padding with zeros.
    y = y.to_tensor()

    return X, y

def make_batches(ds):
  return (
      ds
      .cache()
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE)
      .prefetch(tf.data.AUTOTUNE))



train,val,test = generateDataset()
tokenizer = Tokenizer(['<SEP>','<SYL>','<SOV>','<EOV>','[START]','[END]'],'../outputs/danteVocabulary.txt')
train_batches = make_batches(train)
val_batches = make_batches(val)


#positional encoding
n, d = 2048, 512
pos_encoding = positional_encoding(n, d)
pos_encoding = pos_encoding[0]

'''
# plot positional encoding
pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
pos_encoding = tf.reshape(pos_encoding, (d, n))

plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()
'''

#hyperparameters

transformer_config = {'num_layers': 4,
                      'd_model': 256,   #128
                      'num_heads': 4,   #8
                      'dff': 1024,      #512
                      'dropout_rate': 0.1}

learning_rate = CustomSchedule(transformer_config['d_model'])

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

#define transformer
