# LSTM Text Generators Using Keras

This project describes two different approaches and implimentations to making an LSTM text generater using the Python Library Keras. The first approach used a model that generated one character at a time (the char-based model) and another model that generated one word at a time (the word-based model). For context, these models were intended to be hosted on a Javascript webserver which could be queried as an API and provide generated text. 

The first approach used a char-based model that generated one character at a time. This character would then be appended to the end of the seed text and fed back into the model to predict the next character. This process was very time consuming. The code for building these models and instructions for how to use them can be found in the folder titled "char-based".

In an attempt to speed up the text generation, the second approach used a word-based model that generated full words at a time. This model worked very similarly to the one above. It would generate a word, append it to the seed text, and  attempt to predict the next word. Whenever the model encounters a word it has not seen before, it seems to skip over that word. This means that it may be necessary to add some level of buffer before the provided seed text so that the model will get seed text of the expected length or greater.

## The Char-Based Text Generator

This char-based model is based on the Keras example project given [here](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py).

For more detail, see the python notebook in the char-based folder.

## The Word-Based Text Generator

The word-based is model is based on the work of Jason Brownlee given [here](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/).

For more detail, see the python [notebook](https://github.com/tomiyee/lstm-text-gen/blob/master/word-based/01-Word-Based-LSTM.ipynb) in the word-based folder. 

## Maximum Likelihood Character Level Language Model

This model doesn't use machine learning to generate text. Instead, it scans the entire corpus and tracks the frequency of any given character going after the previous *n* characters. The larger the value of *n*, the more the generated text will resemble the original corpus. A useful property of this model is that the words that it generates will almost always be spelled correctly with an appropriately large value *n*. 

For more detail, see this wonderfully written [notebook](https://nbviewer.jupyter.org/gist/yoavg/d76121dfde2618422139?utm_content=bufferefcf2&utm_medium=social&utm_source=plus.google.com&utm_campaign=buffer) by Yoav Goldberg. 


## Converting Keras Models to Tensorflow.js Models

For information on how to convert exported Keras models to Tensorflow.js models, see the instructions given on the tensorflow website [here](https://www.tensorflow.org/js/tutorials/conversion/import_keras):

In short, you first need to install using pip the tensorflowjs library.

```
pip install tensorflowjs
```

Next, you simply run the following command.

```
tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir
```
In order to load these models into a Javascript file, you will need the following code.
```
import * as tf from '@tensorflow/tfjs';
const model = await tf.loadLayersModel('https://foo.bar/tfjs_artifacts/model.json');
```
