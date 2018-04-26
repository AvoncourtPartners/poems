import tensorflow as tf
import tensor2tensor as tt
import model as m 
import numpy as np
from pathlib import Path
from tensor2tensor.utils import trainer_lib
from tensor2tensor import problems
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import problem


import typing as t 

data_dir = Path("t2t/data")
tmp_dir = Path("t2t/tmp")
train_dir = Path("t2t/train")
checkpoint_dir = Path("t2t/checkpoints")

@registry.register_problem
class PoetryChars(text_problems.Text2TextProblem):
    """Predict next line of poetry from the last line. From Gutenberg texts."""

    @property
    def approx_vocab_size(self):
        return 128

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def vocab_type(self):
        """What kind of vocabulary to use.

        `VocabType`s:
        * `SUBWORD`: `SubwordTextEncoder`, an invertible wordpiece vocabulary.
            Must provide `self.approx_vocab_size`. Generates the vocabulary based on
            the training data. To limit the number of samples the vocab generation
            looks at, override `self.max_samples_for_vocab`. Recommended and
            default.
        * `CHARACTER`: `ByteTextEncoder`, encode raw bytes.
        * `TOKEN`: `TokenTextEncoder`, vocabulary based on a file. Must provide a
            vocabulary file yourself (`TokenTextEncoder.store_to_file`) because one
            will not be generated for you. The vocab file should be stored in
            `data_dir/` with the name specified by `self.vocab_filename`.

        Returns:
        VocabType constant
        """
        return text_problems.VocabType.CHARACTER

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]



run_config=trainer_lib.create_run_config()


hparams = trainer_lib.create_hparams(
    hparams_set = "transformer_tiny", 
    data_dir=data_dir, 
    problem_name="poetry_chars")

estimator = trainer_lib.create_estimator('transformer',hparams,run_config)

def char_ids_gen(poem_config):
    def gen():
        char_gen = m.char_gen(poem_config)()
        char_list = m.get_char_list(poem_config)
        while True:
            char = next(char_gen)
            ind = None
            try:
                ind = char_list.index(char)
            except ValueError:
                ind = 0
            yield ind
    return gen


def tt_input_fn(
        token_generator: t.Callable[[],t.Generator[int,None,None]], 
        hyper_params: dict
    ) -> tf.data.Dataset:
    tokens = tf.data.Dataset.from_generator(token_generator, output_types=tf.int32, output_shapes=())
    one_token_window = tokens.apply(
            m.sliding_window_batch(2)
        ).map(
            lambda w: ({
                "inputs":  tf.reshape(w[0],[1,1,1]),
                "targets": tf.reshape(w[1],[1,1,1])
            })
        )
    window = one_token_window.batch(hyper_params['seq_len'])
    window_r = window.batch(1) # basically a reshape
    prefetch = window.prefetch(buffer_size=1)
    return prefetch

def train():
    return estimator.train(lambda: tt_input_fn(char_ids_gen(m.poem_config), m.hyper_params))


def text_to_ids(text: str, poem_config:dict):
    char_list = m.get_char_list(poem_config)
    def char_to_id(char: str):
        ind = None
        try:
            ind = char_list.index(char)
        except ValueError:
            ind = 0
        return ind
    return list(map(char_to_id,list(text)))

def ids_to_text(list_of_ids: list, poem_config):
    char_list = m.get_char_list(poem_config)
    return "".join(map(lambda i: char_list[i], list_of_ids))

def generate(estimator, poem_config):
    seed_text = "Привет"
    seed_ids  = text_to_ids(seed_text, poem_config)
    seed_ids_ar = np.array(seed_ids).reshape(-1,1,1,1)

    pred_gen = estimator.predict(lambda: tf.data.Dataset.from_tensor(seed_ids, output_types={"inputs": tf.int32}))       
