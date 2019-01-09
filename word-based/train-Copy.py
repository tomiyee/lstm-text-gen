from keras.models import load_model
import string
from numpy import array

import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dropout

from pickle import load
from pickle import dump

import numpy as np
import random
from random import randint
import sys
import io, getopt, ast
from pathlib import Path

# ===================================================================
# parameters

# Path to the raw corpus
raw_corpus ="../datasets/harry-potter-1.txt"
# This is the file with the pre separated lines of 51 words
dataset_path = "./harry-potter.txt"

load_existing_model = False
filename  = "generated-model"
load_path = "./" + filename + ".h5"
save_path = "./"

num_epochs = 5
checkpoints = list(range(1,num_epochs+1))
batch_size = 256
words_to_generate = 60

input_size = 50
output_size = 1

have_period = True
have_comma = True
whitelisted_punctuation = '.,'
# ===================================================================
# Takes Command line inputs to override the above
if __name__ == "__main__":
    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv,"hd:c:e:l:n:",["dataset=","checkpoints=","epochs=","load_model=","name="])
    except getopt.GetoptError:
        print ('train-model.py -d <dataset> -c <epochs to checkpoint> -e <# epochs> -l <model to load from> -n <name of files>')
        sys.exit(2)

    for opt, arg in opts:

        # Help Command
        if opt == '-h':
            print ('train-model.py -d <dataset> -c <epochs to checkpoint> -e <# epochs> -l <model to load from> -n <name of files>')
            sys.exit()

        # Num Epochs
        elif opt in ("-e","--epochs"):
            num_epochs = ast.literal_eval(arg)
            checkpoints = list(range(num_epochs))

        # Load Model
        elif opt in ("-l", "--load_model"):
            load_existing_model = True
            filename = arg
            
        # Name of the file with the raw corpus
        elif opt in ("-d", "--dataset"):
            raw_corpus = "../datasets/" + arg

        # Checkpoints
        elif opt in ("-c", "--checkpoints"):
            checkpoints = ast.literal_eval(arg)

        elif opt in ("-n", "--name"):
            filename = arg
# ===================================================================
# Load The Model or create a new model
if load_existing_model:
    print("Loading Existing Model...")
    # load the model
    model = load_model(filename + '.h5')
    # load the tokenizer
    tokenizer = load(open(filename + '-tokenizer.pkl', 'rb'))
else:
    print("Generating New Model...")
    
    # =================================================
    # Dataset Acquisition
    
    # loads doc into memory
    def load_doc(filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text

    # turns a doc into clean tokens
    def clean_doc(doc):
        # make lower case
        doc = doc.lower()
        # replace '--' with a space ' '
        doc = doc.replace('--', ' ')
        doc = doc.replace('-', ' ')
        for char in whitelisted_punctuation:
            doc = doc.replace('{} '.format(char), ' {} '.format(char))
        # split into tokens by white space
        tokens = doc.split()
        punctuation = string.punctuation
        for char in whitelisted_punctuation:
            punctuation = punctuation.replace(char,'')
        # remove punctuation from each token
        table = str.maketrans('', '', punctuation)
        tokens = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha() or (word in whitelisted_punctuation and word is not "")]
        return tokens

    # load document
    doc = load_doc(raw_corpus)
    # clean document
    tokens = clean_doc(doc)
    print(tokens[:20])
    print('Total Tokens: %d' % len(tokens))
    print('Unique Tokens: %d' % len(set(tokens)))
    
    # =================================================
    # Dataset Preparation and Preservation

    # organize into sequences of tokens
    length = input_size + output_size
    sequences = list()
    for i in range(length, len(tokens)):
        # select sequence of tokens
        seq = tokens[i-length:i]
        # convert into a line
        line = ' '.join(seq)
        # store
        sequences.append(line)
    print('Total Sequences: %d' % len(sequences))

    # save tokens to file, one dialog per line
    def save_doc(lines, filename):
        data = '\n'.join(lines)
        file = open(filename, 'w')
        file.write(data)
        file.close()

    # save sequences to file
    save_doc(sequences, filename + "-lines.txt")
    
    # =================================================
    # Tokenize Lines, Vocab Size Determination
    
    # load doc into memory
    def load_doc(filename):
        # open the file as read only
        file = open(filename, 'r')
        # read all text
        text = file.read()
        # close the file
        file.close()
        return text

    # load
    doc = load_doc(filename + "-lines.txt")
    lines = doc.split('\n')
    
    # integer encode sequences of words
    punctuation = string.punctuation
    for char in whitelisted_punctuation:
        punctuation = punctuation.replace(char,'')
    tokenizer = Tokenizer(filters=punctuation)
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)
    # save the tokenizer
    dump(tokenizer, open(filename + '-tokenizer.pkl', 'wb'))
    
    # vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    print(vocab_size)
    
    # separate into input and output
    sequences = array(sequences)
    X, y = sequences[:,:-1], sequences[:,-1]
    y = to_categorical(y, num_classes=vocab_size)
    seq_length = X.shape[1]
    
    # =================================================
    # Model Creation

    # define model
    model = Sequential()
    model.add(Embedding(vocab_size, input_size, input_length=seq_length))
    model.add(LSTM(96, return_sequences=True))
    model.add(LSTM(96))
    model.add(Dense(96, activation='relu'))
    model.add(Dense(vocab_size, activation='softmax'))
    print(model.summary())
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
# ===================================================================
# Load the Dataset with the lines of text

def load_doc(filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

doc = load_doc(filename + "-lines.txt")
lines = doc.split('\n')

# ===================================================================
# Use the tokenizer we just loaded to prepare the sequences we're using

# load the tokenizer
tokenizer = load(open(filename + '-tokenizer.pkl', 'rb'))
sequences = tokenizer.texts_to_sequences(lines)

# remove punctuation from each token
# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print("Vocab Size: %d" % vocab_size)

# separate into input and output
sequences = array(sequences)

X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# ===================================================================
# Define function to define the generated text

def generate_text():
    
    result = list()
    # select a seed text
    seed_text = lines[randint(0,len(lines))]
    
    for i in range(words_to_generate):
        # encode the seed text
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)

        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break

        # append to input
        seed_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

# ===================================================================
# Define the Callback Function

def on_epoch_end (epoch, _):
    
    # Checkpointing the model
    for i in checkpoints:
        if epoch + 1 == i:
            print("Checkpointing the model...")
            model.save("%s-cp-%d.h5" % (filename, i))
            break
    print("Generating Text...")
    print(generate_text())
    
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
# ===================================================================
# Fit Model

model.fit(X, y, batch_size=batch_size, epochs=num_epochs, callbacks=[print_callback])