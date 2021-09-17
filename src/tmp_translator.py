import tensorflow as tf


def evaluate(enc_input, enc_output, transformer, tokenizer):
    start, end = tokenizer.tokenize([''])[0]
    output = tf.convert_to_tensor([start])
    output = tf.expand_dims(output, 0)


evaluate(None, None, transformer=None,
         tokenizer=tf.keras.preprocessing.text.Tokenizer(lower=True, char_level=False, filters=''))
