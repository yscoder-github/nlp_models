#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
We use following lines because we are running on Google Colab
If you are running notebook on a local computer, you don't need these
"""
from google.colab import drive
drive.mount('/content/gdrive')
import os
os.chdir('/content/gdrive/My Drive/finch/tensorflow2/text_classification/imdb/main')


# In[2]:


get_ipython().system('pip install tensorflow-gpu==2.0.0-alpha0')


# In[3]:


import tensorflow as tf
import numpy as np
import pprint
import logging
import array
import hashlib
import time

print("TensorFlow Version", tf.__version__)
print('GPU Enabled:', tf.test.is_gpu_available())


# In[ ]:


def hashed_ngrams(unigrams, unigram_vocab_size, n_buckets, n):
    """hashed n-gram features."""

    def get_ngrams(n):
        return list(zip(*[unigrams[i:] for i in range(n)]))

    def hash_ngram(ngram):
        bytes_ = array.array('L', ngram).tobytes()
        hash_ = int(hashlib.sha256(bytes_).hexdigest(), 16)
        return unigram_vocab_size + hash_ % n_buckets

    return [hash_ngram(ngram) for ngram in get_ngrams(n)]


def data_generator(f_path, params):
  with open(f_path) as f:
    print('Reading', f_path)
    for line in f:
      line = line.rstrip()
      label, text = line.split('\t')
      text = text.split(' ')
      x = [params['word2idx'].get(w, len(word2idx)) for w in text]
      if len(x) > params['max_len']:
        x = x[:params['max_len']]
      x += hashed_ngrams(x,
                         params['vocab_size'],
                         params['num_buckets'],
                         n=2,)
      y = int(label)
      yield x, y


def dataset(is_training, params):
  _shapes = ([None], ())
  _types = (tf.int32, tf.int32)
  _pads = (0, -1)
  
  if is_training:
    ds = tf.data.Dataset.from_generator(
      lambda: data_generator(params['train_path'], params),
      output_shapes = _shapes,
      output_types = _types,)
    #ds = ds.shuffle(params['num_samples'])
    ds = ds.padded_batch(params['batch_size'], _shapes, _pads)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  else:
    ds = tf.data.Dataset.from_generator(
      lambda: data_generator(params['test_path'], params),
      output_shapes = _shapes,
      output_types = _types,)
    ds = ds.padded_batch(params['batch_size'], _shapes, _pads)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
  
  return ds


# In[ ]:


class FastText(tf.keras.Model):
  def __init__(self, params):
    super().__init__()
    self.unigram = tf.Variable(np.load('../vocab/word.npy'), dtype=tf.float32, name='embedding_unigram')
    self.bigram = tf.Variable(np.random.uniform(size=(params['num_buckets'], 300)), dtype=tf.float32, name='embedding_bigram')
    self.out_linear = tf.keras.layers.Dense(2)
    
  def call(self, inputs):
    if inputs.dtype != tf.int32:
      inputs = tf.cast(inputs, tf.int32)
    masks = tf.cast(tf.sign(inputs), tf.float32)
    
    x = tf.nn.embedding_lookup(tf.concat((self.unigram, self.bigram), axis=0), inputs)
    
    masked_mean = lambda x, mask: tf.reduce_sum(x * tf.expand_dims(mask, -1), 1) / tf.reduce_sum(mask, 1, keepdims=True)
    x = masked_mean(x, masks)
    
    x = self.out_linear(x)
    
    return x


# In[ ]:


params = {
  'vocab_path': '../vocab/word.txt',
  'train_path': '../data/train.txt',
  'test_path': '../data/test.txt',
  'num_samples': 25000,
  'num_labels': 2,
  'batch_size': 32,
  'max_len': 500,
  'num_buckets': int(8E5),
  'num_patience': 3,
  'lr': 3e-4,
}


# In[ ]:


def is_descending(history: list):
  history = history[-(params['num_patience']+1):]
  for i in range(1, len(history)):
    if history[i-1] <= history[i]:
      return False
  return True    


# In[8]:


word2idx = {}
with open(params['vocab_path']) as f:
  for i, line in enumerate(f):
    line = line.rstrip()
    word2idx[line] = i
params['word2idx'] = word2idx
params['vocab_size'] = len(word2idx) + 1


model = FastText(params)
model.build(input_shape=(None, None))
pprint.pprint([(v.name, v.shape) for v in model.trainable_variables])

decay_lr = tf.optimizers.schedules.ExponentialDecay(params['lr'], 1000, 0.95)
optim = tf.keras.optimizers.Adam(params['lr'])
global_step = 0

history_acc = []
best_acc = 0.

t0 = time.time()
logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)


while True:
  # TRAINING
  for texts, labels in dataset(is_training=True, params=params):
    with tf.GradientTape() as tape:
      logits = model(texts)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
      loss = tf.reduce_mean(loss)

    optim.lr.assign(decay_lr(global_step))
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    if global_step % 50 == 0:
      logger.info("Step {} | Loss: {:.4f} | Spent: {:.1f} secs | LR: {:.6f}".format(global_step, loss.numpy().item(), time.time()-t0, optim.lr.numpy().item()))
      t0 = time.time()
    global_step += 1
  
  # EVALUATION
  m = tf.keras.metrics.Accuracy()

  for texts, labels in dataset(is_training=False, params=params):
    logits = model(texts)
    y_pred = tf.argmax(logits, axis=-1)
    m.update_state(y_true=labels, y_pred=y_pred)
    
  acc = m.result().numpy()
  logger.info("Evaluation: Testing Accuracy: {:.3f}".format(acc))  
  history_acc.append(acc)
  
  if acc > best_acc:
    best_acc = acc
    # you can save model here
  logger.info("Best Accuracy: {:.3f}".format(best_acc))
  
  if len(history_acc) > params['num_patience'] and is_descending(history_acc):
    logger.info("Testing Accuracy not improved over {} epochs, Early Stop".format(params['num_patience']))
    break

