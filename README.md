# lstm-text-gen
This project contains a code for building an LSTM Text Generator using the Python Library Keras. These
models were then compiled into a tensorflow.js model that was then hosted on a web-server. This web
server would use the model to generate text given seed text.

The first approach used a model that generated one character at a time. This character would
then be appended to the end of the seed text and fed back into the model to predict the next
character. This process was very time consuming, taking about 3 seconds for every character.
The code for building these models and instructions for how to use them can be found in the
folder titled "char-based."

The second approach used a model that generated entire words at a time. This model worked very
similarly to the one above. It would generate a word, append it to the seed text, and then
attempt to predict the next word given the updated seed text. Whenever the model encounters
a word it has not seen before, it seems to skip over that word. This means that it may be
necessary to add some level of buffer before the provided seed text so that the model will
get seed text of the expected length or greater.

Much of the code seen in this repository is written in Jupyter Notebook files. As such, it is
advised that some python notebook view

## The Character-Based Text Generator from [Original Source]{https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py}

The original character-based generator



## The Word-Based Text Generator
