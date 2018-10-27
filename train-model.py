'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io, getopt, ast
from pathlib import Path

# ===========================================================

# parameters
dataset_path = "./datasets/nancy-drew-1-secret-of-old-clock.txt"

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
       print ('test.py -i <inputfile> -o <outputfile>')
       sys.exit(2)

   for opt, arg in opts:

       # Help Command
       if opt == '-h':
           print ('test.py -l <pathtosaved> -s <pathtosave>')
           sys.exit()

       # Num Epochs
       elif opt in ("-e","--epochs"):
           num_epochs = ast.literal_eval(arg)
           checkpoints = list(range(num_epochs))

       # Dataset Name
       elif opt in ("-d", "--dataset"):
           dataset_path = "./datasets/" + arg

       # Checkpoints
       elif opt in ("-c", "--checkpoints"):
           checkpoints = ast.literal_eval(arg)

       # Load Model
       elif opt in ("-l", "--load_model"):
           load_file = True
           load_file = arg

       elif opt in ("-n", "--name"):
           file_name = arg

# ===========================================================

with io.open(dataset_path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# Outputs a charset
charset = sorted(list(set(text)))
# Make char's JS Readable
for i in range(len(charset)):
    if (charset[i] == '\n'):
        charset[i] = '\\n'
    if (charset[i] == '"'):
        charset[i] = '\\"'
# Generates the charset file
f = open(file_name + "-charset.txt","w+")
charset_final = '["'+ '","'.join(charset) + '"]'
f.write(charset_final)
f.close()

print (charset_final)

# ===========================================================

# cut the text in semi-redundant sequences of maxlen characters
maxlen = look_back
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step_size):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# ===========================================================

# build the model: a single LSTM
print('Build model...')

my_file = Path(load_path)
if load_file and my_file.is_file():
    print("Found Checkpoint. Loading Saved model")
    model = load_model(load_path)
else:
    model = Sequential()
    model.add(LSTM(128, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars), activation='softmax'))

    optimizer = RMSprop(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

# ===========================================================

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# ===========================================================

def on_epoch_end(epoch, _):

    # Checkpointing the model
    for i in checkpoints:
        if epoch + 1 == i:
            print("Checkpointing the model...")
            model.save("%s-%d.h5" % (file_name,i))
    
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

# ===========================================================

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# ===========================================================



model.fit(x, y,
          batch_size=batch_size,
          epochs=num_epochs,
          callbacks=[print_callback])

print("Source: \"%s\" \nEpochs: %d \nBatch Size: %d \nStep Size: %d \n Look Back: %d" % (dataset_path, num_epochs, batch_size, step_size, maxlen))

# ===========================================================
