<img src="https://github.com/zhedongzheng/finch/blob/master/nlp-models/assets/transformer.png" width="300">

<img src="https://github.com/zhedongzheng/finch/blob/master/nlp-models/assets/multihead_attn.png" width='700'>

Some functions are adapted from [Kyubyong's](https://github.com/Kyubyong/transformer) work, thanks for him!

* Based on that, we have:
    * implemented the model under the high level ```TF Estimator``` API

    * added an option to share the weights between encoder embedding and decoder embedding

    * added an option to share the weights between decoder embedding and output projection

    * added the learning rate variation according to the formula in paper, and also expotential decay

    * added more activation choices (leaky relu / elu) for easier gradient propagation

    * fixed a mistake of masking discovered [here](https://github.com/Kyubyong/transformer/issues/3)

    * used ```tf.while_loop``` to perform autoregressive decoding on graph, instead of ```feed_dict```

<img src="https://github.com/zhedongzheng/finch/blob/master/nlp-models/assets/transform20fps.gif" height='400'>
