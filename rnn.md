## RNN
<image src="Figs/rnn/01_rnn.png" hight=300>


## LSTM
http://colah.github.io/posts/2015-08-Understanding-LSTMs/

A special kind of RNN, capable of learning long-term dependencies.

In the last few years, there have been incredible success applying RNNs to a variety of problems: 
speech recognition, language modeling, translation, image captioning.
The list goes on. 
The amazing feats one can achieve with RNNs.
( Andrej Karpathy’s excellent blog post, The Unreasonable Effectiveness of Recurrent Neural Networks)

One of the appeals of RNNs is the idea that they might be able to connect previous information to the present task, 
such as using previous video frames might inform the understanding of the present frame. 
If RNNs could do this, they’d be extremely useful. But can they? It depends.

It’s entirely possible for the gap between the relevant information and the point where it is needed to become very large.


