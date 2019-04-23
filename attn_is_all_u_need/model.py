import tensorflow as tf
from config import args
from modules import *

__author__ = 'yscoder@foxmail.com'


def forward_pass(sources, targets, params, reuse=False):
    with tf.variable_scope('forward_pass', reuse=reuse):
        pos_enc = _get_position_encoder()

        # ENCODER
        en_masks = tf.sign(sources)

        with tf.variable_scope('encoder_embedding', reuse=reuse):
            encoded = embed_seq(
                sources, params['source_vocab_size'], args.hidden_units, zero_pad=True, scale=True)

        with tf.variable_scope('encoder_position_encoding', reuse=reuse):
            encoded += pos_enc(sources, en_masks, args.hidden_units)

        with tf.variable_scope('encoder_dropout', reuse=reuse):
            encoded = tf.layers.dropout(encoded, args.dropout_rate, training=(not reuse))

        for i in range(args.num_blocks):
            with tf.variable_scope('encoder_attn_%d' % i, reuse=reuse):
                encoded = multihead_attn(queries=encoded, keys=encoded, q_masks=en_masks, k_masks=en_masks,
                                         num_units=args.hidden_units, num_heads=args.num_heads,
                                         dropout_rate=args.dropout_rate,
                                         future_binding=False, reuse=reuse, activation=None)

            with tf.variable_scope('encoder_feedforward_%d' % i, reuse=reuse):
                encoded = pointwise_feedforward(encoded, num_units=[4 * args.hidden_units, args.hidden_units],
                                                activation=params['activation'])

        # DECODER
        decoder_inputs = _shift_right(targets, params['start_symbol'])
        de_masks = tf.sign(decoder_inputs)

        if args.tied_embedding:
            with tf.variable_scope('encoder_embedding', reuse=True):
                decoded = embed_seq(decoder_inputs, params['target_vocab_size'], args.hidden_units,
                                    zero_pad=True, scale=True)
        else:
            with tf.variable_scope('decoder_embedding', reuse=reuse):
                decoded = embed_seq(
                    decoder_inputs, params['target_vocab_size'], args.hidden_units, zero_pad=True, scale=True)

        with tf.variable_scope('decoder_position_encoding', reuse=reuse):
            decoded += pos_enc(decoder_inputs, de_masks, args.hidden_units)

        with tf.variable_scope('decoder_dropout', reuse=reuse):
            decoded = tf.layers.dropout(decoded, args.dropout_rate, training=(not reuse))

        for i in range(args.num_blocks):
            with tf.variable_scope('decoder_self_attn_%d' % i, reuse=reuse):
                decoded = multihead_attn(queries=decoded, keys=decoded, q_masks=de_masks, k_masks=de_masks,
                                         num_units=args.hidden_units, num_heads=args.num_heads,
                                         dropout_rate=args.dropout_rate,
                                         future_binding=True, reuse=reuse, activation=None)

            with tf.variable_scope('decoder_attn_%d' % i, reuse=reuse):
                decoded = multihead_attn(queries=decoded, keys=encoded, q_masks=de_masks, k_masks=en_masks,
                                         num_units=args.hidden_units, num_heads=args.num_heads,
                                         dropout_rate=args.dropout_rate,
                                         future_binding=False, reuse=reuse, activation=None)

            with tf.variable_scope('decoder_feedforward_%d' % i, reuse=reuse):
                decoded = pointwise_feedforward(decoded, num_units=[4 * args.hidden_units, args.hidden_units],
                                                activation=params['activation'])

        # OUTPUT LAYER    
        if args.tied_proj_weight:
            b = tf.get_variable('bias', [params['target_vocab_size']], tf.float32)
            _scope = 'encoder_embedding' if args.tied_embedding else 'decoder_embedding'
            with tf.variable_scope(_scope, reuse=True):
                shared_w = tf.get_variable('lookup_table')
            decoded = tf.reshape(decoded, [-1, args.hidden_units])  # -1 is like None
            logits = tf.nn.xw_plus_b(decoded, tf.transpose(shared_w), b)
            logits = tf.reshape(logits, [tf.shape(sources)[0], -1, params['target_vocab_size']])
        else:
            with tf.variable_scope('output_layer', reuse=reuse):
                logits = tf.layers.dense(decoded, params['target_vocab_size'], reuse=reuse)
        return logits


def _model_fn_train(features, mode, params, logits):
    """
    :param features:
    :param mode:
    :param params:
    :param logits:
    :return:
    """
    with tf.name_scope('backward'):
        targets = features['target']
        masks = tf.to_float(tf.not_equal(targets, 0))

        if args.label_smoothing:
            loss_op = label_smoothing_sequence_loss(
                logits=logits, targets=targets, weights=masks, label_depth=params['target_vocab_size'])
        else:
            loss_op = tf.contrib.seq2seq.sequence_loss(
                logits=logits, targets=targets, weights=masks)

        if args.lr_decay_strategy == 'noam':
            step_num = tf.train.get_global_step() + 1  # prevents zero global step
            lr = _get_noam_lr(step_num)
        elif args.lr_decay_strategy == 'exp':
            lr = tf.train.exponential_decay(1e-3, tf.train.get_global_step(), 100000, 0.1)
        else:
            raise ValueError("lr decay strategy must be one of 'noam' and 'exp'")
        log_hook = tf.train.LoggingTensorHook({'lr': lr}, every_n_iter=100)

        train_op = tf.train.AdamOptimizer(lr).minimize(loss_op, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss_op, train_op=train_op, training_hooks=[log_hook])


def _model_fn_predict(features, mode, params):
    def cond(i, x, temp):
        return i < args.target_max_len

    def body(i, x, temp):
        logits = forward_pass(features['source'], x, params, reuse=True)
        ids = tf.argmax(logits, -1)[:, i]
        ids = tf.expand_dims(ids, -1)

        temp = tf.concat([temp[:, 1:], ids], -1)

        x = tf.concat([temp[:, -(i + 1):], temp[:, :-(i + 1)]], -1)
        x = tf.reshape(x, [tf.shape(temp)[0], args.target_max_len])
        i += 1
        return i, x, temp

    _, res, _ = tf.while_loop(cond, body, [tf.constant(0), features['target'], features['target']])

    return tf.estimator.EstimatorSpec(mode=mode, predictions=res)


def tf_estimator_model_fn(features, labels, mode, params):
    logits = forward_pass(features['source'], features['target'], params)
    if mode == tf.estimator.ModeKeys.TRAIN:
        _ = forward_pass(features['source'], features['target'], params, reuse=True)
        return _model_fn_train(features, mode, params, logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return _model_fn_predict(features, mode, params)


def _shift_right(targets, start_symbol):
    """
    :param targets:
    :param start_symbol:
    :return:
    """
    start_symbols = tf.cast(tf.fill([tf.shape(targets)[0], 1], start_symbol), tf.int64)
    return tf.concat([start_symbols, targets[:, :-1]], axis=-1)


def _get_position_encoder():
    if args.position_encoding == 'non_param':
        pos_enc = sinusoidal_position_encoding
    elif args.position_encoding == 'param':
        pos_enc = learned_position_encoding
    else:
        raise ValueError("position encoding has to be either 'param' or 'non_param'")
    return pos_enc


def _get_noam_lr(step_num):
    return tf.rsqrt(tf.to_float(args.hidden_units)) * tf.minimum(
        tf.rsqrt(tf.to_float(step_num)),
        tf.to_float(step_num) * tf.convert_to_tensor(args.warmup_steps ** (-1.5)))
