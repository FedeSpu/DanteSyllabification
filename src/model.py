from src.transformer_utils.custom_schedule import *
from src.transformer import *
from src.transformer_utils.checkpoint import *
import time
import random
import re

EPOCHS = 50

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


class ModelTransformer(object):
    def __init__(self, config, input_vocab_size, target_vocab_size):
        self.learning_rate = CustomSchedule(config['d_model'])
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # LOSS AND METRICS
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')

        self.transformer = Transformer(
            num_layers=config['num_layers'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            dff=config['dff'],
            input_vocab_size=input_vocab_size,
            target_vocab_size=target_vocab_size,
            pe_input=1000,
            pe_target=1000,
            rate=0.1)

        self.cpkt_manager = create_checkpoint(self.transformer, self.optimizer)

    @tf.function(input_signature=train_step_signature)
    def train_step(self, inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = self.transformer([inp, tar_inp], training=True)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(accuracy_function(tar_real, predictions))

    def train(self, train, val, EPOCHS):
        for epoch in range(EPOCHS):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            # added these two
            self.val_loss.reset_states()
            self.val_accuracy.reset_states()

            # inp -> X, tar -> Y
            for (batch, (inp, tar)) in enumerate(train):
                self.train_step(inp=inp, tar=tar)

                if batch % 50 == 0:
                    print(
                        f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} '
                        f'Accuracy {self.train_accuracy.result():.4f}')

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.cpkt_manager.save()
                print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

            # added the part for the validation dataset
            # TODO:cambiarla leggermente
            # -----
            for val_entry in val:
                val_inp = val_entry[0]
                val_tar = val_entry[1]
                val_tar_inp = val_tar[:, :-1]
                val_tar_real = val_tar[:, 1:]
                with tf.GradientTape() as tape:
                    predictions, _ = self.transformer([val_inp, val_tar_inp], training=False)
                loss = loss_function(val_tar_real, predictions)
                self.val_loss(loss)
                self.val_accuracy(accuracy_function(val_tar_real, predictions))
            # -----

            print(f'Epoch {epoch + 1} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')
            print(
                f'Epoch {epoch + 1} Validation loss {self.val_loss.result():.4f} Validation accuracy {self.val_accuracy.result():.4f}')
            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

    def get_transformer(self):
        return self.transformer

    def syllabify(self, sentence, tokenizer):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]

        sentence = tokenizer.tokenize(sentence).to_tensor()
        encoder_input = sentence
        print(encoder_input)
        start_end = tokenizer.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]
        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)
        for i in tf.range(50):
            output = tf.transpose(output_array.stack())
            predictions, _ = self.transformer([encoder_input, output], training=False)
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predicted_id = tf.argmax(predictions, axis=-1)
            output_array = output_array.write(i + 1, predicted_id[0])
            if predicted_id == end:
                break
        output = tf.transpose(output_array.stack())
        # output.shape (1, tokens)
        text = tokenizer.detokenize(output)[0]
        text = text.numpy().decode('utf-8')
        text = re.sub(rf'\[', '', text)
        text = re.sub(rf'\]', '', text)
        return text

    def generate(self, sentence, tokenizer):
        assert isinstance(sentence, tf.Tensor)
        if len(sentence.shape) == 0:
            sentence = sentence[tf.newaxis]
        sentence = tokenizer.tokenize(sentence).to_tensor()
        encoder_input = sentence
        start_end = tokenizer.tokenize([''])[0]
        start = start_end[0][tf.newaxis]
        end = start_end[1][tf.newaxis]

        output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
        output_array = output_array.write(0, start)
        output = tf.transpose(output_array.stack())
        # first prediction
        predictions, _ = self.transformer([encoder_input, output], training=False)
        predictions = predictions[:, -1:, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        output_array = output_array.write(1, predicted_id[0])
        for i in tf.range(1, 50):
            # other predictions
            output = tf.transpose(output_array.stack())
            predictions, _ = self.transformer([encoder_input, output], training=False)
            predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
            predictions = tf.nn.softmax(predictions, axis=-1)
            top = tf.math.top_k(predictions.numpy()[0][0], k=1)  # greed search
            predicted_id = random.choices(top.indices.numpy(), weights=top.values.numpy(), k=1)
            predicted_id = tf.convert_to_tensor([predicted_id], dtype=tf.int64)
            output_array = output_array.write(i + 1, predicted_id[0])
            if predicted_id == end:
                break
        output = tf.transpose(output_array.stack())
        # output.shape (1, tokens)
        text = tokenizer.detokenize(output)[0]
        text = text.numpy().decode('utf-8')
        text = re.sub(rf'\[', '', text)
        text = re.sub(rf'\]', '', text)
        return text
