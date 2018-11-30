from keras.models import load_model
from pickle import load
import string
from numpy import array

import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.callbacks import LambdaCallback
from keras.preprocessing.sequence import pad_sequences

from pickle import dump

import numpy as np
import random
import sys
import io, getopt, ast
from pathlib import Path

# ===================================================
# parameters

dataset_path = "./harry-potter.txt"

load_file = False
load_path = "./checkpoint.h5"
save_path = "./"
file_name = "checkpoint"

num_epochs = 5
checkpoints = list(range(num_epochs))
batch_size = 256
look_back = 40
step_size = 1


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

        # Dataset Name
        elif opt in ("-d", "--dataset"):
            dataset_path = arg

        # Checkpoints
        elif opt in ("-c", "--checkpoints"):
            checkpoints = ast.literal_eval(arg)

        # Load Model
        elif opt in ("-l", "--load_model"):
            load_file = True
            load_file = arg

        elif opt in ("-n", "--name"):
            file_name = arg

# ===================================================
# Load The Model

# load the model
model = load_model('model.h5')
# load the tokenizer
tokenizer = load(open('tokenizer.pkl', 'rb'))

# ===================================================
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
in_filename = 'harry-potter.txt'
doc = load_doc(in_filename)
lines = doc.split('\n')

# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print("Vocab Size: %d" % vocab_size)

# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]


# ===================================================
# Define the Callback Function

def on_epoch_end (epoch, _):
    
    # Checkpointing the model
    for i in checkpoints:
        if epoch + 1 == i:
            print("Checkpointing the model...")
            model.save("%s.h5" % (file_name))
    print("Generating Text...")
    print(generate_text())
    
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)


# ===================================================

from random import randint

def generate_text():
    
    result = list()
    # select a seed text
    seed_text = lines[randint(0,len(lines))]
    
    for i in range(40):
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


# ===================================================
# Fit Model
model.fit(X, y, batch_size=batch_size, epochs=num_epochs, callbacks=[print_callback])
