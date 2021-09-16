from src.transformer_utils.custom_schedule import *
from src.tranformer import *
from src.transformer_utils.checkpoint import *
import time


class ModelTransformer(object):
    def __init__(self, config, tokenizer, input_vocab_size, target_vocab_size):
        super(ModelTransformer, self).__init__()
        self.tokenizer = tokenizer
        # OPTIMIZER
        self.learning_rate = CustomSchedule(config['d_model'])
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        '''
        temp_learning_rate_schedule = CustomSchedule(d_model)
        plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
        plt.ylabel("Learning Rate")
        plt.xlabel("Train Step")
        '''

        # LOSS & METRICS
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

        # TRAINING & CHECKPOINT
        # TODO: Are those important?
        '''
        input_vocab_size=tokenizer.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizer.en.get_vocab_size().numpy(),
        '''
        self.transformer = Transformer(
            num_layers=config['num_layers'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            dff=config['dff'],
            pe_input=1000,
            pe_target=1000,
            rate=config['dropout_rate'],
            input_vocab_size=input_vocab_size,
            target_vocab_size=target_vocab_size)

        self.cpkt_manager = create_checkpoint(self.transformer, self.optimizer)

    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]

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

    def train(self, batches, EPOCHS):
        for epoch in range(EPOCHS):
            start = time.time()

            self.train_loss.reset_states()
            self.train_accuracy.reset_states()

            # inp -> portuguese, tar -> english
            for (batch, (inp, tar)) in enumerate(batches):
                print(type(inp))
                self.train_step(inp=inp, tar=tar)

                if batch % 50 == 0:
                    print(
                        f'Epoch {epoch + 1} Batch {batch} Loss {self.train_loss.result():.4f} '
                        f'Accuracy {self.train_accuracy.result():.4f}')

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print(f'Saving checkpoint for epoch {epoch + 1} at {ckpt_save_path}')

            print(f'Epoch {epoch + 1} Loss {self.train_loss.result():.4f} Accuracy {self.train_accuracy.result():.4f}')

            print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')
