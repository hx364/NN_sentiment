# NN_sentiment
An implementation of CNN/RNN for sentiment classification. This repo has three models: two on sentence level, one on document level.


##Requirment
* python 2.7
* Theano
* keras
* keras-extra

## Model

### 1) sentence-level CNN
The same model as Yoon's [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) A embedding layer followed by convolution layer.

### 2) sentence-level LSTM
Similar to model 1, concatenating a embedding layer with a LSTM-RNN module.

### 3) document-level CNN-LSTM
Implement sentiment classification on document level. The basic idea is to stack CNN and a LSTM. The first layer is a embedding layer initialized by word2vec, transform each word to word embedding representaion. Then a convolution layer to learn a fixed-length representation for each sentence. Then input the sentence-level representaion to a RNN module(GRU/LSTM) for sentiment classification.

## Data
* IMDB movie data for sentence-level
* Yelp review data for document-level

## Result
1) See Yoon's paper for the performance of Model 1
2) Model 3 reaches 70.2% accuracy in 5-class classification on yelp-2015 dataset.
