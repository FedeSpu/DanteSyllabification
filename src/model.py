from src.transformer_utils.custom_schedule import *
from src.transformer_utils.loss import *
from src.tranformer import *
from src.transformer_utils.train_checkpoint import *


def model(config, tokenizer):
    # OPTIMIZER
    learning_rate = CustomSchedule(config['d_model'])
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    '''
    temp_learning_rate_schedule = CustomSchedule(d_model)
    plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    '''

    # LOSS & METRICS
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    # TRAINING & CHECKPOINT
    # TODO: Are those important?
    '''
    input_vocab_size=tokenizer.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizer.en.get_vocab_size().numpy(),
    '''
    transformer = Transformer(
        num_layers=config['num_layers'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        dff=config['dff'],
        pe_input=1000,
        pe_target=1000,
        rate=config['dropout_rate'])

    cpkt_manager = create_checkpoint(transformer, optimizer)
