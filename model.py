import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import typing as t
from tensorflow.contrib.data import sliding_window_batch


tf.logging.set_verbosity(tf.logging.DEBUG)

# Define the character to int mapping
char_list  = [chr(0x0500)] # First character represens all out-of-vocabulary characters
char_list += ['\n']
char_list += list(map(chr,range(0x0020,0x007F))) # Basic Latin unicode block
char_list += list(map(chr,range(0x00A0,0x00FF))) # Latin-1 Supplement unicode block
char_list += list(map(chr,range(0x0400,0x04FF))) # Cyrillic unicode block

def char_line_breaker(line: str)->t.List[str]:
    "Splits a line of text in to a list of characters"
    return list(line)

def token_generator(text_filename: Path, line_breaker:t.Callable[[str],t.List[str]]) -> t.Generator[str,None,None]:
    "Returns a generator that reads a utf-8 encoded text file line by line and yeilds tokens"
    fh = text_filename.open('r', encoding='utf-8')
    line = fh.readline()
    while line != '':
        for token in char_line_breaker(line):
            yield token
        line = fh.readline()
    return None

def input_fn(
        token_generator: t.Callable[[],t.Generator[str,None,None]], 
        hyper_params: dict
    ) -> tf.data.Dataset:
    tokens = tf.data.Dataset.from_generator(token_generator, output_types=tf.string, output_shapes=())
    one_token_window = tokens.apply(sliding_window_batch(2)).map(lambda w: ({"token":w[0]}, w[1]))
    window = one_token_window.batch(hyper_params['seq_len'])
    return window


def create_feature_columns(hyper_params: dict):
    cat = tf.feature_column.categorical_column_with_vocabulary_list(
        key = "token",
        vocabulary_list = char_list,
        default_value = 0)
    embedding = tf.feature_column.embedding_column(cat,hyper_params['embedding_dimention'])
    return [embedding]

def poems_moden_fn(
        features: dict, # This is batch_features from input_fn
        labels: tf.Tensor,   # This is batch_labels from input_fn
        mode,     # An instance of tf.estimator.ModeKeys
        params: dict
    ) -> tf.estimator.EstimatorSpec:  # Additional configuration
    hyper_params = params['hyper_params']

    input_t = tf.feature_column.input_layer(features,params['feature_columns'])
    input_r_t = tf.expand_dims(input_t,0) # Add dimention to create a batch_size of 1 for dynamic_rnn
    layer1_cell = tf.contrib.rnn.BasicLSTMCell(hyper_params['LSTM1_size'], state_is_tuple = False)
    layer1_prev_state = tf.Variable(
        initial_value = layer1_cell.zero_state(1,dtype=tf.float32),
        trainable=False
    )

    layer1_out_t, layer1_state_t = tf.nn.dynamic_rnn(layer1_cell, input_r_t, sequence_length=[hyper_params['seq_len']], initial_state=layer1_prev_state)
    
    state_update_op = layer1_prev_state.assign(layer1_state_t)
    with tf.control_dependencies([state_update_op]):
        logits_t = tf.layers.dense(layer1_out_t[0], len(char_list))
    
    predicted_token_ids = tf.argmax(logits_t,1)
    char_list_t = tf.constant(char_list, dtype=tf.string)
    predicted_tokens    = tf.gather(char_list_t, predicted_token_ids)

    ####################################################################################
    # PREDICT
    ####################################################################################
        
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_token_ids,
            'predicted_tokens': predicted_tokens,
            'probabilities': tf.nn.softmax(logits_t),
            'logits': logits_t,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    ####################################################################################
    # EVAL
    ####################################################################################

    char_map_t = tf.contrib.lookup.HashTable(
        initializer = tf.contrib.lookup.KeyValueTensorInitializer(
            char_list_t,
            tf.range(1,len(char_list)+1, dtype = tf.int32)),
        default_value = 0
        )

    label_ids = char_map_t.lookup(labels)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=label_ids, logits=logits_t)
    accuracy, accuracy_op = tf.metrics.accuracy(labels=labels, predictions = predicted_tokens, name='acc_op')


    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, 
            loss=loss, 
            eval_metric_ops={
                'accuracy': (accuracy, accuracy_op)
                }
            )

    ####################################################################################
    # TRAIN
    ####################################################################################

    optimizer = tf.train.AdagradOptimizer(learning_rate=hyper_params['learning_rate'])
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return None
    



def log_dir_name(hyper_params: dict)->str:
    params = [key + "_" + str(hyper_params[key]) for key in hyper_params]
    timestamp = pd.Timestamp.now()
    return "-".join(params) + "/" + str(int(timestamp.timestamp()))


def create_estimator(hyper_params: dict)-> tf.estimator.Estimator:
    estimator = tf.estimator.Estimator(
        model_fn = poems_moden_fn, 
        model_dir='logs/' + log_dir_name(hyper_params),
        
        config=tf.estimator.RunConfig(
            save_checkpoints_steps = 1000,
            log_step_count_steps   = 1000,
            save_summary_steps     = 10
        ),
        params = { 
            "feature_columns" : create_feature_columns(hyper_params),
            "hyper_params": hyper_params
        }
    )
    return estimator


hyper_params = {
        "embedding_dimention": 5,
        "seq_len": 32,
        "LSTM1_size": 32,
        "learning_rate": 0.1,
    }

def char_gen():
    return token_generator(Path('train_data/Pushkin.txt'), char_line_breaker)

estimator = create_estimator(hyper_params)
estimator.train(lambda: input_fn(char_gen, hyper_params))



