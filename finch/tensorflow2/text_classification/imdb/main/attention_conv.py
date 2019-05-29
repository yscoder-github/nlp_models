#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
We use following lines because we are running on Google Colab
If you are running notebook on a local computer, you don't need these
"""
from google.colab import drive
drive.mount('/content/gdrive')
import os
os.chdir('/content/gdrive/My Drive/finch/tensorflow2/text_classification/imdb/main')


# In[ ]:


get_ipython().system('pip install tf-nightly-2.0-preview')


# In[ ]:


import tensorflow as tf
import numpy as np
import pprint
import logging
import time

print("TensorFlow Version", tf.__version__)
print('GPU Enabled:', tf.test.is_gpu_available())


# In[ ]:


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
    ds = ds.shuffle(params['num_samples'])
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


class KernelAttentivePooling(tf.keras.Model):
  def __init__(self, params):
    super().__init__()
    self.dropout = tf.keras.layers.Dropout(params['dropout_rate'])
    self.kernel = tf.keras.layers.Conv1D(filters=1,
                                         kernel_size=params['kernel_size'],
                                         padding='same',
                                         activation=tf.tanh,
                                         use_bias=False)

  
  def call(self, inputs, training=False):
    x, masks = inputs
    # alignment
    align = tf.squeeze(self.kernel(self.dropout(x, training=training)), -1)
    # masking
    paddings = tf.fill(tf.shape(align), float('-inf'))
    align = tf.where(tf.equal(masks, 0), paddings, align)
    # probability
    align = tf.nn.softmax(align)
    align = tf.expand_dims(align, -1)
    # weighted sum
    return tf.squeeze(tf.matmul(x, align, transpose_a=True), -1)


# In[ ]:


class Model(tf.keras.Model):
  def __init__(self, params):
    super().__init__()
    
    self.embedding = tf.Variable(np.load('../vocab/word.npy'),
                                 dtype=tf.float32,
                                 name='pretrained_embedding')
    
    self.attentive_pooling = KernelAttentivePooling(params)
    
    self.out_linear = tf.keras.layers.Dense(2)
  
  
  def call(self, inputs, training=False):
    if inputs.dtype != tf.int32:
      inputs = tf.cast(inputs, tf.int32)
    masks = tf.sign(inputs)
    
    x = tf.nn.embedding_lookup(self.embedding, inputs)
    
    x = self.attentive_pooling((x, masks), training=training)
    
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
  'max_len': 1000,
  'dropout_rate': 0.2,
  'kernel_size': 5,
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


# In[ ]:


word2idx = {}
with open(params['vocab_path']) as f:
  for i, line in enumerate(f):
    line = line.rstrip()
    word2idx[line] = i
params['word2idx'] = word2idx
params['vocab_size'] = len(word2idx) + 1


model = Model(params)
model.build(input_shape=(None, None))
pprint.pprint([(v.name, v.shape) for v in model.trainable_variables])

decay_lr = tf.optimizers.schedules.ExponentialDecay(params['lr'], 1000, 0.95)
optim = tf.optimizers.Adam(params['lr'])
global_step = 0

history_acc = []
best_acc = .0

t0 = time.time()
logger = logging.getLogger('tensorflow')
logger.setLevel(logging.INFO)


while True:
  # TRAINING
  for texts, labels in dataset(is_training=True, params=params):
    with tf.GradientTape() as tape:
      logits = model(texts, training=True)
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
      loss = tf.reduce_mean(loss)
    
    optim.lr.assign(decay_lr(global_step))
    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    if global_step % 50 == 0:
      logger.info("Step {} | Loss: {:.4f} | Spent: {:.1f} secs | LR: {:.6f}".format(
          global_step, loss.numpy().item(), time.time()-t0, optim.lr.numpy().item()))
      t0 = time.time()
    global_step += 1
  
  # EVALUATION
  m = tf.keras.metrics.Accuracy()

  for texts, labels in dataset(is_training=False, params=params):
    logits = model(texts, training=False)
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

