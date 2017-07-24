import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, recurrent, Embedding
from keras.layers.recurrent import LSTM
from keras.optimizers import Adam, RMSprop
from nltk import FreqDist
import os
import datetime

#read data
source = 'input.txt'
input_file = open(source, 'r')
data = input_file.read()
input_file.close()

x = data.split('\n')

#create an array of every sentence
#convert "how are you" => ["how","are","you"]
text_sequence = [text_to_word_sequence(y, lower=True, split=" ") for y in x]

#create a vocab list with frequency of every word
#set the longest word allowed to be 19 => (0 based)
vocab = FreqDist(np.hstack(text_sequence)).keys()

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
# add padding="post" to add the pads after the content of the array, instead of at first
padded_text_sequence = pad_sequences(text_sequence, padding='post', maxlen=longest_text_length, dtype='int32')

#modeling time

def create_model(vocab_len, max_len, hidden_size, num_layers):
    model = Sequential()
    
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
input_max_len = max([len(sentence) for sentence in padded_text_sequence])
vocab_len = len(vocab)+2
model = create_model(vocab_len, input_max_len, hidden_dim, layer_num)

def process_data(word_sentences, max_len, word_to_index):
    # Vectorizing each element in each sequence
    sequences = np.zeros((len(word_sentences), max_len, len(word_to_index)))
    for i, sentence in enumerate(word_sentences):
        for j, word in enumerate(sentence):
            sequences[i, j, word] = 1.
    return sequences



# questions = padded_text_sequence[0:][::2]
# answers = padded_text_sequence[1:][::2]
# answer_sequences = process_data(answers, len(answers[0]), word_to_index)

# model.fit(questions,
#     answer_sequences, 
#     batch_size=int(len(questions)/10),
#     epochs=20, 
#     verbose=1, 
#     callbacks=None, 
#     validation_split=0.0, 
#     validation_data=None, 
#     shuffle=True, 
#     class_weight=None, 
#     sample_weight=None, 
#     initial_epoch=0)

# model.save_weights('output/checkpoint_epoch_{}.hdf5'.format(1))

questions = padded_text_sequence[0:][::2]
answers = padded_text_sequence[1:][::2]

# Training 10 sequences at a time
# i_end = 0
# print(len(answers[0]))
# for k in range(1, 11):
    
#     # Shuffling the training data every epoch to avoid local minima
#     indices = np.arange(len(questions))
#     np.random.shuffle(indices)
#     questions = questions[indices]
#     answers = answers[indices]
    
#     for i in range(0, len(questions), 10):
#         if i + 10 >= len(questions):
#             i_end = len(questions)
#         else:
#             i_end = i + 10
#         answer_sequences = process_data(answers[i:i_end], len(answers[0]), word_to_index)
#         model.fit(questions[i:i_end],answer_sequences, verbose=1, epochs=1)
#     model.save_weights('output/checkpoint_epoch_{}.hdf5'.format(k))


def find_checkpoint_file(folder):
    checkpoint_file = [f for f in os.listdir(folder) if 'checkpoint' in f]
    if len(checkpoint_file) == 0:
        return []
    modified_time = [os.path.getmtime(f) for f in checkpoint_file]
    return checkpoint_file[np.argmax(modified_time)]

# saved_weights = find_checkpoint_file('output')
model.load_weights('output/checkpoint_epoch_10.hdf5')

predictions = np.argmax(model.predict(questions), axis=2)
sequences = []
questions_text = x[0:][::2]

for i, prediction in enumerate(predictions):
    sequence = ' '.join([index_to_word[index] for index in prediction if index > 0])
    print(sequence)
    sequences.extend([questions_text[i], sequence])

np.savetxt('output/test_result.txt', sequences, fmt='%s')