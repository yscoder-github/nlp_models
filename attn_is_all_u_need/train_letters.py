import json

import tensorflow as tf
from config import args
from data import DataLoader
from model import tf_estimator_model_fn
from utils import greedy_decode, prepare_params

tf.logging.set_verbosity(tf.logging.INFO)


def main():
    dl = DataLoader(
        source_path='../temp/letters_source.txt',
        target_path='../temp/letters_target.txt')
    sources, targets = dl.load()

    tf_estimator = tf.estimator.Estimator(
        tf_estimator_model_fn, params=prepare_params(dl))

    for epoch in range(6):
        tf_estimator.train(tf.estimator.inputs.numpy_input_fn(
            x={'source': sources, 'target': targets},
            batch_size=args.batch_size,
            num_epochs=None,
            shuffle=True), steps=1000)
        greedy_decode(['apple', 'common', 'zhedong'], tf_estimator, dl)


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()
