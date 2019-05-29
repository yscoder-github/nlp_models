"""
@author: yinshuai
该例子比较简单,因为所使用的输入样本集的长度都很短,所以在训练的时候的配置如下:

skip_window_size: 2
batch_size: 20
train_sample:
    input: target word (当前窗口的中心词) 每个单词都会作为中心词
    output: word in windows (i - 1) & (i + 1),这里只选择窗口中的一个单词作为训练输出



"""
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

tf.reset_default_graph()

# 3 Words Sentence
sentences = ["i like dog", "i like cat", "i like animal",
             "dog cat animal", "apple cat dog like", "dog fish milk like",
             "dog cat eyes like", "i like apple", "apple i hate",
             "apple i movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split()  # 将上面的sentence数组合并
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}  # vocabulary 词汇表

# Word2Vec Parameter
batch_size = 20
embedding_size = 2  # To show 2 dim embedding graph
voc_size = len(word_list)


def random_batch(data, size):
    """
    @author: yinshuai
    ### 下面是胡扯呀
    word2vector有如下两种主流方式:
    (1)Skip-Gram:
     训练时的输入与输出如下:
      input:  __ __ good __
      output: this is  a
      而训练的时候虽然窗口大小选定了,但是一般也不会使用窗口中的全部单词,而是会随机选择一部分
         作为训练所用输出
      举例:
        input: good
        output:
         batch:
            first_train_output:  ['this']
            second_train_output: ['this','is']
            third_train_output: ['a']
    :param data: Skip-Gram当前单词的周边时间窗口的数个单词构成的数组
    :param size: (batch_size)
    :return:

    ### 下面才是该代码中的解释:
    :param data: 由于当前训练窗口的大小为2,而data中记录着[target[dim=1], context word[dim=1]]构成的多维数组
    :size (batch size)
    其实本函数就是为了从所有的可能训练集中随机选择出一部分训练数据,构成一个批次
    """
    random_inputs = []
    random_labels = []
    # 从当前单词的skip-gram窗口中选择出一次训练的批次中各个训练数据样例的随机索引,
    # random_index为一个数组,数组中记录着该批次数据的各条数据所对应的随机截止位置
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(np.eye(voc_size)[data[i][0]])  # target
        random_labels.append(np.eye(voc_size)[data[i][1]])  # context word

    return random_inputs, random_labels


# Make skip gram of one size window
skip_grams = []
for i in range(1, len(word_sequence) - 1):
    target = word_dict[word_sequence[i]]  # 每个单词都会作为当前训练输入词汇
    # 当前训练的输入词汇的上下文会作为输出训练数据(在这里的窗口大小为2)
    context = [word_dict[word_sequence[i - 1]], word_dict[word_sequence[i + 1]]]

    for w in context:
        skip_grams.append([target, w])

# Model
inputs = tf.placeholder(tf.float32, shape=[None, voc_size])
labels = tf.placeholder(tf.float32, shape=[None, voc_size])

# W and WT is not Traspose relationship
W = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
WT = tf.Variable(tf.random_uniform([embedding_size, voc_size], -1.0, 1.0))

hidden_layer = tf.matmul(inputs, W)  # [batch_size, embedding_size]
output_layer = tf.matmul(hidden_layer, WT)  # [batch_size, voc_size]

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=labels))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for epoch in range(5000):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)
        _, loss = sess.run([optimizer, cost], feed_dict={inputs: batch_inputs, labels: batch_labels})

        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

        trained_embeddings = W.eval()

for i, label in enumerate(word_list):
    x, y = trained_embeddings[i]
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()
