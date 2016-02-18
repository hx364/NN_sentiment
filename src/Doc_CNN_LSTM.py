

import cPickle
import numpy as np
import sys
from keras.optimizers import RMSprop, Adadelta
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation, Flatten,Merge, Reshape
from keras.layers.recurrent import LSTM, GRU
from keras.regularizers import l2
from keras.layers.embeddings import Embedding
from sklearn.cross_validation import train_test_split
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import expand_label
from keras.preprocessing.sequence import pad_sequences
from utils import pad_2Dsequences
from nltk.tokenize import sent_tokenize
from sklearn.metrics import accuracy_score, roc_auc_score
from data_prepare import transform_text
from keras.layers.extra import TimeDistributedConvolution2D, TimeDistributedMaxPooling2D, TimeDistributedFlatten
from keras.utils.np_utils import to_categorical
sys.path.append('../../src/')
from data_prepare import *


class text_CNN_GRU:
    def __init__(self, embedding_mat=None, maxlen_doc=7, maxlen_sent=50, filter_length=[3, 4, 5, 6],
                 nb_filters=200, n_vocab=10000, embedding_dims=300, hidden_gru=64, n_classes=5):
        if embedding_mat is not None:
            self.n_vocab, self.embedding_dims = embedding_mat.shape
        else:
            self.n_vocab = n_vocab
            self.embedding_dims = embedding_dims
        self.maxlen_doc = maxlen_doc
        self.maxlen_sent = maxlen_sent
        self.filter_length = filter_length
        self.nb_filters = nb_filters
        self.hidden_gru = hidden_gru

        print "Building the model"
        #graph model
        model=Graph()
        model.add_input(name='input', input_shape=(self.maxlen_doc*self.maxlen_sent,), dtype='int')

        #Model embedding layer, for word index-> word embedding transformation
        model.add_node(Embedding(self.n_vocab, self.embedding_dims, weights=[self.embedding_mat],
                                 input_length=self.maxlen_sent*self.maxlen_doc),
                       name='embedding', input='input')
        model.add_node(Reshape((self.maxlen_doc, 1, self.maxlen_sent, self.embedding_dims)),
                      name='reshape_5d', input='embedding')
        #define the different filters
        conv_layer = []
        for each_length in filter_length:
            model.add_node(TimeDistributedConvolution2D(self.nb_filters/len(filter_length),
                                                        each_length, self.embedding_dims, border_mode='valid',
                                               input_shape=(self.maxlen_doc,1,self.maxlen_sent, self.embedding_dims)),
                          name='conv_{}'.format(each_length), input='reshape_5d')
            model.add_node(Activation('relu'),
                          name='relu_conv_{}'.format(each_length), input='conv_{}'.format(each_length))

            model.add_node(TimeDistributedMaxPooling2D(pool_size=(int(self.maxlen_sent - each_length+1), 1),
                          border_mode='valid'),
                          name='pool_conv_{}'.format(each_length), input='relu_conv_{}'.format(each_length))
            model.add_node(TimeDistributedFlatten(),
                          name='flatten_conv_{}'.format(each_length), input='pool_conv_{}'.format(each_length))
            conv_layer.append('flatten_conv_{}'.format(each_length))
        # model.add_node(Activation('relu'), name='relu', inputs=conv_layer)
        print conv_layer
        model.add_node(GRU(self.hidden_gru), name='gru_forward', inputs=conv_layer)
        model.add_node(GRU(self.hidden_gru, go_backwards=True), name='gru_backward', inputs=conv_layer)
        model.add_node(Dropout(0.5), name='gru_outputs', inputs=['gru_forward', 'gru_backward'])
        model.add_node(Dense(n_classes), name='full_con', input='gru_outputs')
        model.add_node(Activation('softmax'), name='prob', input='full_con')
        model.add_output(name='pred', input='prob')

        model.compile('rmsprop', loss = {'pred': 'categorical_crossentropy'})


    def fit(self, X_train, y_train, X_test, y_test, batch_size=256, nb_epoch=8):
        MODEL_FILE = './graph-test-model.h5'
        self.model.fit(
            {'input': X_train, 'pred': y_train}, batch_size=batch_size, nb_epoch=nb_epoch,
            validation_data=({'input': X_test, 'pred': y_test}),
            callbacks =
                   [
                       EarlyStopping(verbose=True, patience=2, monitor='val_loss'),
                       ModelCheckpoint(MODEL_FILE, monitor='val_loss', verbose=True, save_best_only=True)
                   ])


    def save_weights(self, fname='../data/doc_yelp_data/text_cnn_gru_weights.h5'):
        self.model.save_weights(fname)

    def load_model(self, fname='../data/doc_yelp_data/text_cnn_gru_weights.h5'):
        self.model.load_weights(fname)

    def predict(self, X_test):
        X_pred = np.array(self.model.predict({'input': X_test})['pred'])
        X_pred = np.argmax(X_pred, axis=1)
        return X_pred

    def predict_prob(self, X_test):
        X_pred = np.array(self.model.predict({'input': X_test})['pred'])
        return X_pred

    def accuracy_score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_test = np.argmax(y_test, axis=1)
        return accuracy_score(y_test, y_pred)



def index_to_input(X, maxlen_sent, maxlen_doc):
    """
    transform the index-list based input to some data that can be fed into the text_CNN_GRU class
    :param X: [[word_index]]
    :return:
    """
    X = [pad_sequences(i, maxlen=maxlen_sent) for i in X]
    X = pad_2Dsequences(X, maxlen=maxlen_doc)
    return X



def text_to_input(text, vocabulary, maxlen_sent, maxlen_doc):
    """
    transform a list of texts input to the data that can be fed into the text_CNN_GRU class
    :param text:
    :return:
    """
    text = [sent_tokenize(i) for i in list(text)]
    text = [[clean_str(i) for i in j] for j in list(text)]
    X = [[[vocabulary[word] for word in sentence] for sentence in doc] for doc in text]
    X = index_to_input(X, maxlen_sent, maxlen_doc)
    return X


if __name__ == "__main__":
    print "Loading the data"
    x, y, embedding_mat = cPickle.load(open('../data/doc_yelp_data/data_x_y_w2v.p', 'rb'))
    vocabulary, vocabulary_inv = cPickle.load(open('../data/doc_yelp_data/vocabulary.p', 'rb'))

    print "Train Test split"
    nb_samples, maxlen_doc, maxlen_sent = x.shape
    X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=42)

    print "Preparing the data"
    y_train = np.asarray(y_train, dtype='int32')-1
    y_train = to_categorical(y_train, nb_classes=5)
    y_test = np.asarray(y_test, dtype='int32')-1
    y_test = to_categorical(y_test, nb_classes=5)


    print "Training"
    clf = text_CNN_GRU(embedding_mat=embedding_mat, maxlen_doc=maxlen_doc, maxlen_sent=maxlen_sent)
    clf.fit(X_train, y_train, X_test, y_test, nb_epoch=10)
    # clf.load_model(fname='../data/cnn_diff_filter.h5')

    print "Dumping the model"
    clf.save_weights(fname='../data/text_cnn_gru_weights.h5')

    print "Evaluation on test set"
    print "Accuracy: %.3f" %clf.accuracy_score(X_test, y_test)