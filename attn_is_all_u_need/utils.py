import numpy as np
import tensorflow as tf
from config import args


def greedy_decode(test_words, tf_estimator, dl):
    test_indices = []
    for test_word in test_words:
        test_idx = [dl.source_word2idx[c] for c in test_word] + \
                   [dl.source_word2idx['<pad>']] * (args.source_max_len - len(test_word))
        test_indices.append(test_idx)
    test_indices = np.atleast_2d(test_indices)

    zeros = np.zeros([len(test_words), args.target_max_len], np.int64)

    pred_ids = tf_estimator.predict(tf.estimator.inputs.numpy_input_fn(
        x={'source': test_indices, 'target': zeros}, batch_size=len(test_words), shuffle=False))
    pred_ids = list(pred_ids)

    target_idx2word = {i: w for w, i in dl.target_word2idx.items()}
    for i, test_word in enumerate(test_words):
        ans = ''.join([target_idx2word[id] for id in pred_ids[i]])
        print(test_word, '->', ans.replace('<end>', ''))


def prepare_params(dl):
    if args.activation == 'relu':
        activation = tf.nn.relu
    elif args.activation == 'elu':
        activation = tf.nn.elu
    elif args.activation == 'lrelu':
        activation = tf.nn.leaky_relu
    else:
        raise ValueError("acitivation fn has to be 'relu' or 'elu' or 'lrelu'")
    params = {
        'source_vocab_size': len(dl.source_word2idx),
        'target_vocab_size': len(dl.target_word2idx),
        'start_symbol': dl.target_word2idx['<start>'],
        'activation': activation}
    return params
