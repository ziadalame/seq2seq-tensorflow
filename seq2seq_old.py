
# coding: utf-8


import numpy as np
from keras.preprocessing.text import text_to_word_sequence



from nltk import FreqDist
import os
import datetime



#read data
source = 'input.txt'
input_file = open(source, 'r')
data = input_file.read()
input_file.close()



x = data.split('\n')


text_to_word_sequence(x[2], lower=True, split=" ")



#create an array of every sentence
#convert "how are you" => ["how","are","you"]
text_sequence = [text_to_word_sequence(y, lower=True, split=" ") for y in x]



#create a vocab list with frequency of every word
#set the longest word allowed to be 19 => (0 based)
vocab = FreqDist(np.hstack(text_sequence)).keys()
#vocab



# Creating an array of words from the vocabulary set, we will use this array as index-to-word dictionary
index_to_word = [word for word in vocab]
# Adding the word "ZERO" to the beginning of the array
index_to_word.insert(0, 'ZERO')
# Adding the word 'UNK' to the end of the array (stands for UNKNOWN words)
index_to_word.append('UNK')


# Creating the word-to-index dictionary from the array created above
word_to_index = {word:ix for ix, word in enumerate(index_to_word)}


# Converting each word to its index value
for i, sentence in enumerate(text_sequence):
    for j, word in enumerate(sentence):
        if word in word_to_index:
            text_sequence[i][j] = word_to_index[word]
        else:
            text_sequence[i][j] = word_to_index['UNK']



#get the length of the longest sentence
longest_text_length = max([len(sentence) for sentence in text_sequence])
#pad the sequence
from keras.preprocessing.sequence import pad_sequences
# add padding="post" to add the pads after the content of the array, instead of at first
padded_text_sequence = pad_sequences(text_sequence, padding='post', maxlen=longest_text_length, dtype='int32')



#modeling time
#import necessary modules
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop



def create_model(vocab_len, max_len, hidden_size, num_layers):
    model = Sequential()
    print(vocab_len)
    print(max_len)
    # Creating encoder network
    model.add(Embedding(vocab_len, 1000, input_length=max_len, mask_zero=True))
    model.add(LSTM(hidden_size))
    model.add(RepeatVector(max_len))

    # Creating decoder network
    for _ in range(num_layers):
        model.add(LSTM(hidden_size, return_sequences=True))
    model.add(TimeDistributed(Dense(vocab_len)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
            optimizer='rmsprop',
            metrics=['accuracy'])
    return model



hidden_dim = 1000
layer_num = 3
input_max_len = max([len(sentence) for sentence in text_sequence])
vocab_len = len(vocab)+2
model = create_model(vocab_len, input_max_len, hidden_dim, layer_num)



def process_data(word_sentences, max_len, word_to_index):
    # Vectorizing each element in each sequence
    sequences = np.zeros((len(word_sentences), max_len, len(word_to_index)))
    for i, sentence in enumerate(word_sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1.
    return sequences



questions = padded_text_sequence[0:][::2]
answers = padded_text_sequence[1:][::2]
answer_sequences = process_data(answers, vocab_len, word_to_index)

print(questions)
print(answers)

print(answer_sequences)

model.fit(questions,
    answer_sequences, 
    batch_size=None,
    epochs=20, 
    verbose=0, 
    callbacks=None, 
    validation_split=0.0, 
    validation_data=None, 
    shuffle=True, 
    class_weight=None, 
    sample_weight=None, 
    initial_epoch=0)

#model.save_weights('checkpoint_epoch_{}.hdf5'.format(k))