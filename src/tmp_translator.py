import tensorflow as tf
from src.transformer_utils.masking import *


def choose_greedy(logits):
    # select the last character from the seq_len dimension
    predicted_ids = tf.argmax(logits[:, -1:, :], axis=-1)
    return predicted_ids


class TmpTranslator(tf.Module):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text_input, max_length=10, start='<go>', stop='<eov>'):
        assert isinstance(text_input, tf.Tensor)

        start_symbol = self.tokenizer.word_index[self.start]
        stop_symbol = self.tokenizer.word_index[self.stop]

        encoder_input = tf.convert_to_tensor(text_input)
        decoder_input = tf.repeat([[start_symbol]], repeats=encoder_input.shape[0], axis=0)

        output = decoder_input
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
        enc_output = model.get_transformer().encoder(encoder_input, False, enc_padding_mask)
        # print(encoder_input)
        # p, aw = model.get_transformer().call((encoder_input, output), False)

        for _ in range(10):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
            dec_outuput, _ = model.get_transformer().decoder(output, enc_output, False, combined_mask, dec_padding_mask)
            predictions = model.get_transformer().final_layer(dec_outuput)
            predicted_ids = choose_greedy(predictions)

            output = tf.concat([tf.cast(output, dtype=tf.int64), tf.cast(predicted_ids, dtype=tf.int64), ], axis=1)

        # print(output)
        stripped_output = list(map(lambda x: x.split('<EOV>')[0], tokenizer.sequences_to_texts(output.numpy())))
        # stripped_output = list(map(strip_tokens, stripped_output))
        print(stripped_output)
