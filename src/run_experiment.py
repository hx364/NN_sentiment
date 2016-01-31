__author__ = 'Hao'
import cPickle
from text_sentiment_CNN import TextCNN
from text_sentiment_LSTM import TextLSTM
import pandas as pd
import sys

x, y, embedding_mat = cPickle.load(open('../data/train_mat.pkl'))
vocab = cPickle.load(open('../data/vocab.pkl'))

clf_cnn = TextCNN(embedding_mat=embedding_mat)
clf_cnn.load_model()

clf_lstm = TextLSTM(embedding_mat=embedding_mat)
clf_lstm.load_model()

df = pd.read_csv(sys.args[1])
texts = list(df['text'])

cnn_score = clf_cnn.predict_text(texts, vocab)[:1]
lstm_score = clf_lstm.predict_text(texts, vocab)[:1]

df['cnn_score'] = cnn_score
df['lstm_score'] = lstm_score
df.to_csv('df_sentiment.csv', index=False)