import tensorflow as tf
import numpy as np
from pathlib import Path
import typing as t

tf.logging.set_verbosity(tf.logging.DEBUG)

# Define the character to int mapping
char_list = ['\n']
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
    from tensorflow.contrib.data import sliding_window_batch
    tokens = tf.data.Dataset.from_generator(token_generator, output_types=tf.string, output_shapes=())
    one_token_window = tokens.apply(sliding_window_batch(2)).map(lambda w: ({"token":w[0]}, w[1]))
    window = one_token_window.batch(hyper_params['seq_len'])
    return window.batch(1)


def create_feature_columns(hyper_params: dict):
    cat = tf.feature_column.categorical_column_with_vocabulary_list("token",char_list)
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
    layer1_cell = tf.contrib.rnn.BasicLSTMCell(hyper_params['LSTM1_size'])
    layer1_prev_state = layer1_cell.zero_state(1,dtype=tf.float32)

    layer1_out_t, layer1_state_next_t = tf.nn.dynamic_rnn(input_t, sequence_length=hyper_params['seq_len'], initial_state=layer1_prev_state)

    logits_t = tf.layers.dense(layer1_out_t, len(char_list))


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
        "LSTM1_size": 32
    }

# Test code
def char_gen():
    return token_generator(Path('train_data/Pushkin.txt'),char_line_breaker)
def char_gen2():
    return map(lambda x:chr(x % 200), range(1000_000))
db = input_fn(char_gen, hyper_params)
it = db.make_one_shot_iterator()
sess = tf.Session()
while True:
    try:
        window = sess.run(it.get_next())
        #tf.logging.debug(window)
        b = b"".join(window[0]["token"])
        tf.logging.debug(b.decode())
    except tf.errors.OutOfRangeError:
        break
