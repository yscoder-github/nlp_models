* The code has been run on Google Colab which provides free GPU memory

#### Contents

* [Text Classification（文本分类）](https://github.com/zhedongzheng/finch#text-classification)

* [Text Matching（文本匹配）](https://github.com/zhedongzheng/finch#text-matching)

* [Spoken Language Understanding（对话理解）](https://github.com/zhedongzheng/finch#spoken-language-understanding)

* [Semantic Parsing（语义解析）](https://github.com/zhedongzheng/finch#semantic-parsing)

* [Knowledge Graph Completion（知识图谱补全）](https://github.com/zhedongzheng/finch#knowledge-graph-completion)

* [Question Answering（问题回答）](https://github.com/zhedongzheng/finch#question-answering)

---

## Text Classification

```
└── finch/tensorflow2/text_classification/imdb
	│
	├── data
	│   └── glove.840B.300d.txt          # pretrained embedding, download and put here
	│   └── make_data.ipynb              # step 1. make data and vocab: train.txt, test.txt, word.txt
	│   └── train.txt  		     # incomplete sample, format <label, text1, text2> separated by \t 
	│   └── test.txt   		     # incomplete sample, format <label, text1, text2> separated by \t
	│
	├── vocab
	│   └── word.txt                     # incomplete sample, list of words in vocabulary
	│	
	└── main              
		└── attention_linear.ipynb   # step 2: train and evaluate model
		└── attention_conv.ipynb     # step 2: train and evaluate model
		└── fasttext_unigram.ipynb   # step 2: train and evaluate model
		└── fasttext_bigram.ipynb    # step 2: train and evaluate model
		└── sliced_rnn.ipynb         # step 2: train and evaluate model
```

* Task: [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/)
	
	* [\<Notebook>: Make Data & Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/data/make_data.ipynb)
		
		* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/data/train.txt)
	
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/vocab/word.txt)

	* Model: TF-IDF + Logistic Regression
	
		* PySpark
		
			* [\<Notebook> TF-IDF + Logistic Regression -> 88.2% Testing Accuracy](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/spark/text_classification/imdb/tfidf_lr.ipynb)

	* Model: [FastText](https://arxiv.org/abs/1607.01759)
	
		* [Facebook Official Release](https://github.com/facebookresearch/fastText)
		
			* [\<Notebook> Unigram FastText -> 87.3% Testing Accuracy](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/framework/official_fasttext/text_classification/imdb/unigram.ipynb)
		
			* [\<Notebook> Bigram FastText -> 89.8% Testing Accuracy](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/framework/official_fasttext/text_classification/imdb/bigram.ipynb)

		* TensorFlow 2

			* [\<Notebook> Unigram FastText -> 89.1 % Testing Accuracy](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/fasttext_unigram.ipynb)
				
			* [\<Notebook> Bigram FastText -> 90.2 % Testing Accuracy](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/fasttext_bigram.ipynb)
	
	* Model: [Feedforward Attention](https://arxiv.org/abs/1512.08756)

		* TensorFlow 2

			* [\<Notebook> Unigram Alignment -> 89.5 % Testing Accuracy](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/attention_linear.ipynb)
			
			* [\<Notebook> N-gram Alignment -> 90.7 % Testing Accuracy](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/attention_conv.ipynb)
	
	* Model: [Sliced RNN](https://arxiv.org/abs/1807.02291)

		* TensorFlow 2

			* [\<Notebook> Sliced RNN -> 91.3 % Testing Accuracy](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_classification/imdb/main/sliced_rnn.ipynb)

---

## Text Matching

```
└── finch/tensorflow2/text_matching/snli
	│
	├── data
	│   └── glove.840B.300d.txt       # pretrained embedding, download and put here
	│   └── download_data.ipynb       # step 1. run this to download snli dataset
	│   └── make_data.ipynb           # step 2. run this to generate train.txt, test.txt, word.txt 
	│   └── train.txt  		  # incomplete sample, format <label, text1, text2> separated by \t 
	│   └── test.txt   		  # incomplete sample, format <label, text1, text2> separated by \t
	│
	├── vocab
	│   └── word.txt                  # incomplete sample, list of words in vocabulary
	│	
	└── main              
		└── dam.ipynb      	  # step 3. train and evaluate model
		└── esim.ipynb      	  # step 3. train and evaluate model
```

* Task: [SNLI](https://nlp.stanford.edu/projects/snli/)

	* [\<Notebook>: Download Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/data/download_data.ipynb)
	
	* [\<Notebook>: Make Data & Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/data/make_data.ipynb)
		
		* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/data/train.txt)
		
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/vocab/word.txt)

	* Model: [DAM](https://arxiv.org/abs/1606.01933)
	
		* TensorFlow 2
		
			* [\<Notebook> -> 85.3% Testing Accuracy](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/dam.ipynb)

	* Model: [ESIM](https://arxiv.org/abs/1609.06038)
	
		* TensorFlow 2
		
			* [\<Notebook> -> 87.4% Testing Accuracy](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/text_matching/snli/main/esim.ipynb)

---

## Spoken Language Understanding

```
└── finch/tensorflow2/spoken_language_understanding/atis
	│
	├── data
	│   └── glove.840B.300d.txt       # pretrained embedding, download and put here
	│   └── make_data.ipynb           # step 1. run this to generate vocab: word.txt, intent.txt, slot.txt 
	│   └── atis.train.w-intent.iob   # incomplete sample, format <text, slot, intent>
	│   └── atis.test.w-intent.iob    # incomplete sample, format <text, slot, intent>
	│
	├── vocab
	│   └── word.txt                  # list of words in vocabulary
	│   └── intent.txt                # list of intents in vocabulary
	│   └── slot.txt                  # list of slots in vocabulary
	│	
	└── main              
		└── bigru.ipynb           # step 2. train and evaluate model
		└── transformer.ipynb     # step 2. train and evaluate model
```

* Task: [ATIS](https://github.com/yvchen/JointSLU/tree/master/data)

	* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/data/atis.train.w-intent.iob)

	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/data/make_data.ipynb)
	
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/vocab/word.txt)

	* Model: [Bi-directional GRU](https://www.ijcai.org/Proceedings/16/Papers/425.pdf)
	
		* TensorFlow 2

			* [\<Notebook> -> 97.4% Intent Accuracy and 95.5% Slot Accuracy on Testing Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/main/bigru.ipynb)

	* Model: [Transformer](https://arxiv.org/abs/1706.03762)
	
		* TensorFlow 2

			* [\<Notebook> -> 97.2% Intent Accuracy and 95.6% Slot Accuracy on Testing Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/spoken_language_understanding/atis/main/transformer.ipynb)

---

## Semantic Parsing

```
└── finch/tensorflow2/semantic_parsing/tree_slu
	│
	├── data
	│   └── glove.840B.300d.txt     	# pretrained embedding, download and put here
	│   └── make_data.ipynb           	# step 1. run this to generate vocab: word.txt, intent.txt, slot.txt 
	│   └── train.tsv   		  	# incomplete sample, format <text, tokenized_text, tree>
	│   └── test.tsv    		  	# incomplete sample, format <text, tokenized_text, tree>
	│
	├── vocab
	│   └── source.txt                	# list of words in vocabulary for source (of seq2seq)
	│   └── target.txt                	# list of words in vocabulary for target (of seq2seq)
	│	
	└── main              
		└── transformer_part1.ipynb 	# step 2. train and evaluate model
		└── transformer_part2.ipynb     # restore from part1 and continue training
		└── transformer_part3.ipynb 	# restore from part2 and continue training
		└── rnn_seq2seq_part1.ipynb     # step 2. train and evaluate model
		└── rnn_seq2seq_part2.ipynb     # restore from part 1 and continue training
```

* Task: [Semantic Parsing for Task Oriented Dialog](https://aclweb.org/anthology/D18-1300)

	* [\<Text File>: Data Example](https://github.com/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/data/train.tsv)

	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/data/make_data.ipynb)
	
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/vocab/target.txt)

	* Model: [Transformer](https://arxiv.org/abs/1706.03762)
	
		* TensorFlow 2
			
			* 71.3% Exact Match Accuracy on Testing Data
			
				* [\<Notebook Part1>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/main/transformer_part1.ipynb) [\<Notebook Part2>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/main/transformer_part2.ipynb) [\<Notebook Part3>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/main/transformer_part3.ipynb)

	* Model: [RNN Seq2Seq](https://arxiv.org/abs/1409.0473)
	
		* TensorFlow 2
			
			* 72.6% Exact Match Accuracy on Testing Data
			
				* [\<Notebook Part1>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/main/rnn_seq2seq_part1.ipynb) [\<Notebook Part2>](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/semantic_parsing/tree_slu/main/rnn_seq2seq_part2.ipynb)

---

## Knowledge Graph Completion

```
└── finch/tensorflow2/knowledge_graph_completion/wn18
	│
	├── data
	│   └── download_data.ipynb       	# step 1. run this to download wn18 dataset
	│   └── make_data.ipynb           	# step 2. run this to generate vocabulary: entity.txt, relation.txt
	│   └── wn18  		          	# wn18 folder (will be auto created by download_data.ipynb)
	│   	└── train.txt  		  	# incomplete sample, format <entity1, relation, entity2> separated by \t
	│   	└── valid.txt  		  	# incomplete sample, format <entity1, relation, entity2> separated by \t 
	│   	└── test.txt   		  	# incomplete sample, format <entity1, relation, entity2> separated by \t
	│
	├── vocab
	│   └── entity.txt                  	# incomplete sample, list of entities in vocabulary
	│   └── relation.txt                	# incomplete sample, list of relations in vocabulary
	│	
	└── main              
		└── distmult_1-N.ipynb    	# step 3. train and evaluate model
```

* Task: WN18

	* [\<Notebook>: Download Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/data/download_data.ipynb)
	
		* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/data/wn18/train.txt)
	
	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/data/make_data.ipynb)
	
		* [\<Text File>: Vocabulary Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/vocab/relation.txt)
		
	* Model: [DistMult](https://arxiv.org/abs/1412.6575) + [1-N Fast Evaluation](https://arxiv.org/abs/1707.01476)

		* TensorFlow 2

			* [\<Notebook> -> 94.9% Hits@10 and 81.0% MRR on Testing Data](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow2/knowledge_graph_completion/wn18/main/distmult_1-N.ipynb)

---

## Question Answering

<img src="https://github.com/DSKSD/DeepNLP-models-Pytorch/blob/master/images/10.dmn-architecture.png" width='500'>

```
└── finch/tensorflow1/question_answering/babi
	│
	├── data
	│   └── make_data.ipynb           		# step 1. run this to generate vocabulary: word.txt 
	│   └── qa5_three-arg-relations_train.txt       # one complete example of babi dataset
	│   └── qa5_three-arg-relations_test.txt	# one complete example of babi dataset
	│
	├── vocab
	│   └── word.txt                  		# complete list of words in vocabulary
	│	
	└── main              
		└── 
```

* Task: [bAbI](https://research.fb.com/downloads/babi/)

	* [\<Text File>: Data Example](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/question_answering/babi/data/qa5_three-arg-relations_test.txt)
	
	* [\<Notebook>: Make Vocabulary](https://nbviewer.jupyter.org/github/zhedongzheng/finch/blob/master/finch/tensorflow1/question_answering/babi/data/make_data.ipynb)
	
	* Model: [Dynamic Memory Network](https://arxiv.org/abs/1603.01417)
