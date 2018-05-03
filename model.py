import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
import typing as t
from tensorflow.contrib.data import sliding_window_batch
import itertools
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs
from tensorflow.python.training import training_util
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer

import jsonlines

tf.logging.set_verbosity(tf.logging.DEBUG)


def char_line_breaker(line: str)->t.List[str]:
    "Splits a line of text in to a list of characters"
    return list(line)

def token_generator(text_filename: Path, line_breaker:t.Callable[[str],t.List[str]]) -> t.Generator[str,None,None]:
    "Returns a generator that reads a utf-8 encoded text file line by line and yeilds tokens"
    tf.logging.info(f"Opening training data from: {text_filename}")
    fh = text_filename.open('r', encoding='utf-8')
    line = fh.readline()
    while line != '':
        for token in char_line_breaker(line):
            yield token
        line = fh.readline()
    return None

def join_tensor(tokens_t: tf.Tensor) -> tf.Tensor:
    "Joins all string in a tokens tensor and returns a scalar tensor with a joined string"
    def join(tokens_np: np.array) -> np.array:
        ba = b''.join(tokens_np)
        return np.array(ba)
    
    return tf.py_func(join,[tokens_t],tf.string, stateful=False)

def input_fn(
        token_generator: t.Callable[[],t.Generator[str,None,None]], 
        hyper_params: dict
    ) -> tf.data.Dataset:
    tokens = tf.data.Dataset.from_generator(token_generator, output_types=tf.string, output_shapes=(None,))
    one_token_window = tokens.apply(sliding_window_batch(2)) 
    # one_token_window value example: 
    # [[b'F', b'd', b's'],
    #  [b'i', b' ', b'e']]
    window = one_token_window.batch(hyper_params['seq_len'])
    window_transpose = window.map(lambda w: ({"token":tf.transpose(w[:,0,:])}, tf.transpose(w[:,1,:]))) 
    # window_transpose value example:
    # ({'token': [['F', 'i', 'r', 's', b]'t'],
    #             ['d', ' ', 'u', 'p', '.'],
    #             ['s', 'e', 'n', 't', '\n']]},
    #  [['i', 'r', 's', 't', ' '],
    #   [' ', 'u', 'p', '.', '\n'],
    #   ['e', 'n', 't', '\n', 'H']])

    packed_as_workaround = window_transpose.map(lambda w0, w1: (
            {"token" : tf.reshape(w0["token"],[-1])},
            tf.reshape(w1,[-1])
        ))

    prefetch = packed_as_workaround.prefetch(buffer_size=1)
    return prefetch


def create_feature_columns(hyper_params: dict, poem_config: dict):
    cat = tf.feature_column.categorical_column_with_vocabulary_list(
        key = "token",
        vocabulary_list = get_char_list(poem_config),
        default_value = 0)
    if hyper_params['embedding_dimention']:
        col = tf.feature_column.embedding_column(cat,hyper_params['embedding_dimention'])
    else:
        col = tf.feature_column.indicator_column(cat)
    return [col]


def count_trainable_params() -> int:
    tv = tf.trainable_variables()
    shapes = map(lambda t:t.shape.as_list(),tv)
    return sum(map(np.prod, shapes))
    

def poems_model_fn(
        features: dict, # This is batch_features from input_fn
        labels: tf.Tensor,   # This is batch_labels from input_fn
        mode,     # An instance of tf.estimator.ModeKeys
        params: dict
    ) -> tf.estimator.EstimatorSpec:  # Additional configuration
    hyper_params = params['hyper_params']
    poem_config  = params['poem_config']
    char_list    = get_char_list(poem_config)
    elem_type = tf.float32
    batch_size   = params['batch_size']
    seq_len      = params['seq_len']

    input_packed_t = tf.feature_column.input_layer(features,params['feature_columns'])
    
    # Work around the inability of feature_colums to support sequnces. Need to manually pack and unpack the tensor 
    input_t = None
    last_unpack_dimention = hyper_params['embedding_dimention'] if hyper_params['embedding_dimention'] else len(char_list)
    input_t = tf.reshape(input_packed_t,(batch_size,seq_len,last_unpack_dimention))


    rnn_sublayer_cells = [
        tf.nn.rnn_cell.LSTMCell(
            size, 
            state_is_tuple = False
        )
        for size in hyper_params['LSTM1_size']]

    rnn_sublayer_cells_dropout = [
        tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob = 1-hyper_params['dropout']) 
        for cell in rnn_sublayer_cells
    ]
    rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_sublayer_cells_dropout, state_is_tuple=False)
    
    rnn_prev_state = tf.Variable(
        initial_value = rnn_cell.zero_state(batch_size, dtype = elem_type),
        trainable=False,
        name='state_memory'
    )
    
    layer1_out_t, rnn_state_t = tf.nn.dynamic_rnn(
        rnn_cell, 
        input_t, 
        sequence_length=list(itertools.repeat(seq_len,batch_size)), 
        initial_state=rnn_prev_state,
        dtype = elem_type
    )

    tf.summary.histogram("rnn_state_t",rnn_state_t)
    tf.summary.histogram("layer1_out_t",layer1_out_t)

    
    state_update_op = rnn_prev_state.assign(rnn_state_t)
    with tf.control_dependencies([state_update_op]):
        logits_t = tf.layers.dense(layer1_out_t, len(char_list))
    
    tf.summary.histogram("logits_t",logits_t)
    
    predicted_token_ids = tf.argmax(logits_t,-1) # -1 means the last dimention
    char_list_t = tf.constant(char_list, dtype = tf.string)
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
            tf.range(0,len(char_list), dtype = tf.int32)),
        default_value = 0
        )
    labels_unpacked = tf.reshape(labels,(batch_size,seq_len))

    label_ids = char_map_t.lookup(labels_unpacked)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=label_ids, logits=logits_t)
    accuracy, accuracy_op = tf.metrics.accuracy(labels=labels_unpacked, predictions = predicted_tokens, name='acc_op')
    perplexity = tf.exp(loss)
    tf.summary.scalar("accuracy", accuracy_op)
    tf.summary.scalar("perplexity", perplexity)

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

    optimizer = None
    if hyper_params["optimizer"] == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=hyper_params['learn_rate'])
    
    if hyper_params["optimizer"] == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate=hyper_params['learn_rate'])
    
    
    grads_and_vars: t.List[t.Tuple[tf.Tensor,tf.Tensor]] = optimizer.compute_gradients(loss)

    for (grad, variable) in grads_and_vars:
        tf.summary.histogram(grad.name, grad)
        tf.summary.histogram(variable.name, variable)

    if hyper_params['grad_clip']:
        clip_value_min = -hyper_params['grad_clip']
        clip_value_max = hyper_params['grad_clip']
        clipped_grads_and_vars = [(tf.clip_by_value(g, clip_value_min, clip_value_max), v) for (g,v) in grads_and_vars]
    else:
        clipped_grads_and_vars = grads_and_vars

    train_op = optimizer.apply_gradients(clipped_grads_and_vars, global_step=tf.train.get_global_step())


    num_of_trainable_params = count_trainable_params()
    tf.logging.info("The number of trainable parameters is: {:,}".format(num_of_trainable_params))


    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return None
    



def log_dir_name(hyper_params: dict, poem_config: dict)->str:
    def val_to_str(val):
        if type(val) == list:
            return "_".join(map(str,val))
        else:
            return str(val)

    params = [key + "_" + val_to_str(hyper_params[key]) for key in hyper_params]
    timestamp = pd.Timestamp.now()
    timestamp_str = "ts" # str(int(timestamp.timestamp()))
    prefix = "gs://checkpt/ml/" if poem_config['use_gs'] else "logs/"
    path = prefix + poem_config['train_set'] + "/" + "-".join(params) + "/" + timestamp_str
    tf.logging.debug(f"Log dir path: {path}")
    return path


def create_estimator(hyper_params: dict, poem_config: dict, predict: bool)-> tf.estimator.Estimator:
    estimator = tf.estimator.Estimator(
        model_fn = poems_model_fn, 
        model_dir=log_dir_name(hyper_params, poem_config),
        
        config=tf.estimator.RunConfig(
            save_checkpoints_steps = None,
            save_checkpoints_secs  = 1200,
            log_step_count_steps   = 1000,
            save_summary_steps     = 100,
            keep_checkpoint_max    = 10,
        ),
        params = { 
            "feature_columns" : create_feature_columns(hyper_params, poem_config),
            "hyper_params": hyper_params,
            "poem_config" : poem_config,
            "batch_size"  : 1 if predict else hyper_params['batch_size'],
            "seq_len"     : 1 if predict else hyper_params['seq_len']
        }
    )
    return estimator


h300 = {
        "embedding_dimention": 5,
        "seq_len": 128,
        "LSTM1_size": [300,300,300],
        "dropout": 0.2,
        "learn_rate": 0.1,
        "optimizer": 'adagrad',
        "grad_clip" : None,
        "batch_size": 1
    }

hyper_params = h300

h500 = {
    'embedding_dimention': 10,
    'seq_len': 128,
    'LSTM1_size': [500, 500, 500, 500],
    'dropout': 0.5,
    "learn_rate": 0.1,
    "optimizer": 'adagrad',
    "grad_clip" : None,
    "batch_size": 1
}

h2_1000 = {
    'embedding_dimention': 5,
    'seq_len': 128,
    'LSTM1_size': [1000, 1000],
    'dropout': 0.5,
    "learn_rate": 0.1,
    "optimizer": 'adagrad',
    "grad_clip" : None,
    "batch_size": 1
}

h1_1000 = {
    'embedding_dimention': 5,
    'seq_len': 128,
    'LSTM1_size': [1000],
    'dropout': 0.5,
    "learn_rate": 0.1,
    "optimizer": 'adagrad',
    "grad_clip" : None,
    "batch_size": 1
}

h2_200 = {
    'embedding_dimention': 5,
    'seq_len': 128,
    'LSTM1_size': [1000,200],
    'dropout': 0.3,
    "learn_rate": 0.1,
    "optimizer": 'adagrad',
    "grad_clip" : None,
    "batch_size": 1
}

h3_512 = {
    'embedding_dimention': 5,
    'seq_len': 128,
    'LSTM1_size': [512,512,512],
    'dropout': 0.3,
    "learn_rate": 0.1,
    "optimizer": 'adagrad',
    "grad_clip" : None,
    "batch_size": 1
}
h3_512_k = {
    'embedding_dimention': None,
    'seq_len': 50,
    'LSTM1_size': [512,512,512],
    'dropout': 0.3,
    "learn_rate": 0.002,
    "optimizer": 'rmsprop',
    "grad_clip" : 5,
    "batch_size": 50
}

poem_config = {
    "use_gs": True,
    "train_set": "pushkin",
    "profile": False
}


train_sets = {
    "goethe": {
        'file_name': 'train_data/Faust_Goethe.txt',
        'char_list': list("\n !'(),-.:;?ABCDEFGHIJKLMNOPRSTUVWZabcdefghijklmnoprstuvwzßäöü")
    },
    "pushkin": {
        'file_name': 'train_data/Pushkin.txt',
        'char_list': list('\t\n !"\'()*,-.1:;<>?[]acdeilmnoprstuv\xa0«»АБВГДЕЖЗИКЛМНОПРСТУФХЧШЭЯабвгдежзийклмнопрстуфхцчшщъыьэюяё–—…')
    },
    "nerudo": {
        'file_name': 'train_data/Pablo_Nerudo.txt',
        'char_list': list('\n !,.:?ACDEHLMNOPQRSTVYabcdefghijlmnopqrstuvxyzáéíñóú')
    },
    "rilke": {
        'file_name': 'train_data/Rilke.txt',
        'char_list': list('\n ,.ABDEGHILMSWabcdefghiklmnoprstuvwzßäöü')
    },
    "shakespeare": {
        'file_name': 'train_data/Shakespeare.txt',
        'char_list': list(" etoahsnri\nldumy,wfcgI:bpA.vTk'SEONRL;CHWMUBD?F!-GPYKVjqxJzQZX")
    }
}

seed_texts = {
    "goethe": 'der Sinn des Lebens',
    "pushkin": 'Жизнь она ведь',
    "nerudo": 'El significado de la vida',
    "rilke": 'der Sinn des Lebens',
    "shakespeare": 'The meaning of life'
}

def get_char_list(poem_config: dict) -> t.List[str]:
    first = chr(0x0500) # First character represens all out-of-vocabulary characters
    char_list = train_sets[poem_config['train_set']]['char_list']
    return [first] + t.cast(t.List[str],char_list) # This cast is to silence a false mypy error


def char_gen(hyper_params = hyper_params, poem_config = poem_config):
    def gen():
        token_gen = token_generator(Path(train_sets[poem_config['train_set']]['file_name']), char_line_breaker)
        all_tokens = np.array(list(token_gen))
        # discard the tail that can not be reshaped to batch sized tensor
        batch_size = hyper_params['batch_size']
        all_tokens_cut = all_tokens[:all_tokens.size-(all_tokens.size % batch_size)]
        batches = all_tokens_cut.reshape((batch_size, -1))
        for i in range(batches.shape[1]):
            yield batches[:,i]
        
    return gen

def char_gen_t1(poem_config = poem_config):
    def gen():
        return itertools.chain.from_iterable(itertools.repeat(list("abcdefghijklmno"),10000))
    return gen

def char_gen_t2(poem_config = poem_config):
    def gen():
        return itertools.chain.from_iterable(itertools.repeat(list("pqrst"),10))
    return gen


# This hook collects profiling information that is used to display compute time in TensorBoard
class MetadataHook(SessionRunHook):
    def __init__ (self,
                  save_steps=None,
                  save_secs=None,
                  output_dir=""):
        self._output_tag = "step-{}"
        self._output_dir = output_dir
        self._timer = SecondOrStepTimer(
            every_secs=save_secs, every_steps=save_steps)

    def begin(self):
        self._next_step = None
        self._global_step_tensor = training_util.get_global_step()
        self._writer = tf.summary.FileWriter (self._output_dir, tf.get_default_graph())

        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use ProfilerHook.")

    def before_run(self, run_context):
        self._request_summary = (
            self._next_step is None or
            self._timer.should_trigger_for_step(self._next_step)
        )
        requests = {"global_step": self._global_step_tensor}
        opts = (tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            if self._request_summary else None)
        return SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results["global_step"]
        global_step = stale_global_step + 1
        if self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)
            self._writer.add_run_metadata(
                run_values.run_metadata, self._output_tag.format(global_step))
            self._writer.flush()
        self._next_step = global_step + 1

    def end(self, session):
        self._writer.close()


def train(hyper_params = hyper_params, poem_config = poem_config):
    estimator = create_estimator(hyper_params, poem_config, predict=False)
    
    profile_hooks=[tf.train.ProfilerHook(
            save_steps=1000,
            output_dir=log_dir_name(hyper_params, poem_config),
            show_memory=True
        ), MetadataHook(
            save_steps=1000,
            output_dir=log_dir_name(hyper_params, poem_config)
        )]
    return estimator.train(
        lambda: input_fn(char_gen(hyper_params, poem_config), hyper_params).skip(1000), 
        hooks=profile_hooks if poem_config['profile'] else None  
    )

def evaluate(hyper_params = hyper_params, poem_config = poem_config):
    estimator = create_estimator(hyper_params, poem_config, predict=False)
    return estimator.evaluate(lambda: input_fn(char_gen(hyper_params, poem_config), hyper_params).take(1000))

def generate_text(
    seed_text: str, 
    num_tokens: int, 
    theta = 4.0, 
    seed = None, 
    hyper_params = hyper_params, 
    poem_config = poem_config,
    checkpoint_path = None):
    "Generates num_tokens chars of text after initializing the LSTMs with the seed_text string"
    composed_list: t.List[str] = []
    processed_seed: t.List[bytes] = []
    full_res_list = []
    char_list = get_char_list(poem_config)
    estimator = create_estimator(hyper_params, poem_config,predict=True)

    

    def char_gen_t3():
        for c in seed_text:
            yield {"token": [[c]]}

        for c in composed_list:
            yield {"token": [[c]]}

    def softmax(x):
        ps = np.exp(x, dtype = np.float64)
        ps /= np.sum(ps)
        return ps

    pred_gen = estimator.predict(
        lambda: tf.data.Dataset.from_generator(char_gen_t3, output_types={"token": tf.string}),
        checkpoint_path = checkpoint_path
        )
    
    for _ in range(len(seed_text)-1):
        pred = next(pred_gen)
        full_res_list.append(pred)
        processed_seed.append(pred['predicted_tokens'][0])

    processed_seed_str = b''.join(processed_seed).decode()

    rs = np.random.RandomState(seed)
    for _ in range(num_tokens):
        pred = next(pred_gen)
        logits = pred['logits'][0]
        probabilities = softmax(logits * theta)
        char_id = rs.choice(probabilities.shape[0],p=probabilities)
        #char_id = np.argmax(probabilities)
        char = char_list[char_id]
        composed_list.append(char)
        full_res_list.append(pred)

    composed_str = ''.join(composed_list)

    return (processed_seed_str, composed_str, full_res_list)

def checkpoint(hyper_params = hyper_params, poem_config = poem_config):
    char_list = get_char_list(poem_config)
    _, gen_text, _ = generate_text(
        seed_text = seed_texts[poem_config['train_set']],
        num_tokens = 1000,
        hyper_params = hyper_params, 
        poem_config = poem_config)
    
    print("Generated text:")
    print(gen_text.replace(char_list[0],"\t"))

    estimator = create_estimator(hyper_params, poem_config, predict=True)

    gen_text_log = Path('logs/generated_text.jsonl')
    gen_text_log.parent.mkdir(parents=True, exist_ok=True)
    with jsonlines.open(gen_text_log, mode='a') as writer:
        writer.write({
            "gen_text": gen_text,
            "walltime": pd.Timestamp.now().isoformat(),
            "global_step": int(estimator.get_variable_value('global_step')),
            "hyper_params": hyper_params,
            "poem_config": poem_config
        })

def run_forever(hyper_params = hyper_params, poem_config = poem_config):
    while True:
        train(hyper_params = hyper_params, poem_config = poem_config)
        evaluate(hyper_params = hyper_params, poem_config = poem_config)
        checkpoint(hyper_params = hyper_params, poem_config = poem_config)

#checkpoint({**h1_1000, "batch_size":1, "seq_len": 32}, {**poem_config,'use_gs':False, 'train_set':'shakespeare'})
