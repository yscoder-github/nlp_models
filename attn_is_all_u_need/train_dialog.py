import json

import tensorflow as tf
from config import args
from data import DataLoader
from model import tf_estimator_model_fn
from utils import greedy_decode, prepare_params

tf.logging.set_verbosity(tf.logging.INFO)


def main():
    dl = DataLoader(
        source_path='../data/dialog_source.txt',
        target_path='../data/dialog_target.txt')
    sources, targets = dl.load()
    print('Source Vocab Size:', len(dl.source_word2idx))
    print('Target Vocab Size:', len(dl.target_word2idx))

    tf_estimator = tf.estimator.Estimator(
        tf_estimator_model_fn, params=prepare_params(dl))

    for epoch in range(1):
        tf_estimator.train(tf.estimator.inputs.numpy_input_fn(
            x={'source': sources, 'target': targets},
            batch_size=args.batch_size,
            num_epochs=1,
            shuffle=True))
        greedy_decode(['你是谁', '你喜欢我吗', '给我唱一首歌', '我帅吗'], tf_estimator, dl)


if __name__ == '__main__':
    print(json.dumps(args.__dict__, indent=4))
    main()
