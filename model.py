import tensorflow as tf
import numpy as np
from pathlib import Path
import typing as t

tf.logging.set_verbosity(tf.logging.DEBUG)

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

def input_fn(token_generator: t.Callable[[],t.Generator[str,None,None]]) -> tf.data.Dataset:
    from tensorflow.contrib.data import sliding_window_batch
    tokens = tf.data.Dataset.from_generator(token_generator, output_types=tf.string, output_shapes=())
    one_token_window = tokens.apply(sliding_window_batch(2)).map(lambda w: ({"token":w[0]}, w[1]))
    window = one_token_window.batch(10)
    return window



# Test code
def char_gen():
    return token_generator(Path('train_data/Pushkin.txt'),char_line_breaker)
def char_gen2():
    return map(lambda x:chr(x % 200), range(1000_000))
db = input_fn(char_gen)
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
