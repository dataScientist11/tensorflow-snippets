"""
Created on Tue Sep 25 11:49:17 2018
@company: Superior Data Science LLC 
@author: bking
"""
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence
import tempfile

class movieReviewData():

#vocab_size = 5000
#embedding_size = 50
#sentence_size=200
#start_id = 1
#oov_id = 2
#index_offset = 2

    def __init__(self):
        
        self.vocab_size = 5000
        self.start_id = 1
        self.oov_id = 2
        self.index_offset = 2
        self.sentence_size = 200
        
        
        model_dir = tempfile.mkdtemp()  
        
        print("Loading data...")
        (self.x_train_variable, self.y_train), (self.x_test_variable, self.y_test) = imdb.load_data(
            num_words=self.vocab_size, start_char=self.start_id, oov_char=self.oov_id,
            index_from=self.index_offset)
        
        self.x_train = 0
        self.x_test = 0
        
        print(len(self.y_train), "train sequences")
        print(len(self.y_test), "test sequences")
        
        
        
    def preProcessing(self):
        '''
            Description: 
                - Load data
                - Convert from text to index
                - 0 post-padding 
            Usage:
        '''
        
#        sentence_size = 200
    #    embedding_size = 50
    
        
        # we assign the first indices in the vocabulary to special tokens that we use
        # for padding, as start token, and for indicating unknown words
        
        pad_id = 0
    
             
        print("Pad sequences (samples x time)")
        self.x_train = sequence.pad_sequences(self.x_train_variable, 
                                         maxlen=self.sentence_size,
                                         truncating='post',
                                         padding='post',
                                         value=pad_id)
        self.x_test = sequence.pad_sequences(self.x_test_variable, 
                                        maxlen=self.sentence_size,
                                        truncating='post',
                                        padding='post', 
                                        value=pad_id)
        
        print("x_train shape:", self.x_train.shape)
        print("x_test shape:", self.x_test.shape)
        

    
    def convert2Text(self,pad_id,oov_id,start_id,index_offset):
        '''
            Description: covert index to text
            Usage:
        '''
        word_index = imdb.get_word_index()
        word_inverted_index = {v + index_offset: k for k, v in word_index.items()}
        
        # The first indexes in the map are reserved to represent things other than tokens
        word_inverted_index[pad_id] = '<PAD>'
        word_inverted_index[start_id] = '<START>'
        word_inverted_index[oov_id] = '<OOV>'
        
        for i in range(0, 10):
          print(i, word_inverted_index[i])
          
        def index_to_text(indexes):
            return ' '.join([word_inverted_index[i] for i in indexes])
        
        print(index_to_text(self.x_train_variable[0]))
    