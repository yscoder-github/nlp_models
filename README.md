## nlp-models 
<p align="center">
<img width="100" src="https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1559145755227&di=eac73643be2de89cc8d0b8d03e79c4f2&imgtype=0&src=http%3A%2F%2Fimg.91jm.com%2F2016%2F08%2F1565A358901.jpg" />
<img width="100" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/225px-TensorFlowLogo.svg.png" />  <img width="100" src="https://avatars1.githubusercontent.com/u/26200681?s=400&u=678028f046e903873ea38f04d23a397dc0d475d8&v=4" /></p>


This [nlp-models](https://github.com/yscoder-github/nlp-models) project is a tutorial for who is studying NLP(Natural Language Processing) using **TensorFlow** , **Pytorch** and **keras** , most of the models in NLP were implemented with less than **200 lines** of code. 

To do: 
- Add some new  model 
- Add keras version 
- Adding more English annotations
- Enriching training examples
- Adding larger data sets to measure model effectiveness



## Catalog

#### 1. Basic Embedding Model

- 1-1. [NNLM(Neural Network Language Model)](https://github.com/yscoder-github/nlp_models/blob/master/1-1.NNLM) - **Predict Next Word**
  - Paper -  [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- 1-2. [Word2Vec(Skip-gram)](https://github.com/yscoder-github/nlp_models/blob/master/1-2.Word2Vec) - **Embedding Words and Show Graph**
  - Paper - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
- 1-3. [FastText(Application Level)](https://github.com/yscoder-github/nlp_models/blob/master/1-3.FastText) - **Sentence Classification**
  - Paper - [Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)


#### 2. CNN(Convolutional Neural Network)

- 2-1. [TextCNN](https://github.com/yscoder-github/nlp_models/blob/master/2-1.TextCNN) - **Binary Sentiment Classification**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
- 2-2. DCNN(Dynamic Convolutional Neural Network)



#### 3. RNN(Recurrent Neural Network)

- 3-1. [TextRNN](https://github.com/yscoder-github/nlp_models/blob/master/3-1.TextRNN) - **Predict Next Step**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
- 3-2. [TextLSTM](https://github.com/yscoder-github/nlp_models/blob/master/3-2.TextLSTM) - **Autocomplete**
  - Paper - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
- 3-3. [Bi-LSTM](https://github.com/yscoder-github/nlp_models/blob/master/3-3.Bi-LSTM) - **Predict Next Word in Long Sentence**




#### 4. Attention Mechanism

- 4-1. [Seq2Seq](https://github.com/yscoder-github/nlp_models/blob/master/4-1.Seq2Seq) - **Change Word**
  - Paper - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
- 4-2. [Seq2Seq with Attention](https://github.com/yscoder-github/nlp_models/blob/master/4-2.Seq2Seq(Attention)) - **Translate**
  - Paper - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
- 4-3. [Bi-LSTM with Attention](https://github.com/yscoder-github/nlp_models/blob/master/4-3.Bi-LSTM(Attention)) - **Binary Sentiment Classification**


#### 5. Model based on Transformer

- 5-1.  [The Transformer](https://github.com/yscoder-github/nlp_models/blob/master/5-1.Transformer) - **Translate**
  - Paper - [Attention Is All You Need(2017)](https://arxiv.org/abs/1810.04805)
- 5-2. [BERT](https://github.com/yscoder-github/nlp_models/blob/master/5-2.BERT) - **Classification Next Sentence & Predict Masked Tokens**
  - Paper - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)


---- 
#### Table presentation

|           Model            |              Example               |   Framework   | 
| :------------------------: | :--------------------------------: | :-----------: |
|            NNLM            |         Predict Next Word          | Torch, Tensor |  
|     Word2Vec(Softmax)      |   Embedding Words and Show Graph   | Torch, Tensor |     
|          TextCNN           |      Sentence Classification       | Torch, Tensor |     
|          TextRNN           |         Predict Next Step          | Torch, Tensor |    
|          TextLSTM          |            Autocomplete            | Torch, Tensor |   
|          Bi-LSTM           | Predict Next Word in Long Sentence | Torch, Tensor |        
|          Seq2Seq           |            Change Word             | Torch, Tensor |   
|   Seq2Seq with Attention   |             Translate              | Torch, Tensor |     
|   Bi-LSTM with Attention   |  Binary Sentiment Classification   | Torch, Tensor |    
|        Transformer         |             Translate              |     Torch     |       
| Greedy Decoder Transformer |             Translate              |     Torch     |     
|            BERT            |            how to train            |     Torch     | 




## Dependencies

- Python 3.5+
- Tensorflow 1.12.0+
- Pytorch 0.4.1+



###  
Plan to add Keras Version   


---- 
#####   Contact Email : yscoder@foxmail.com 