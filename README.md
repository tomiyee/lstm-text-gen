# LSTM Text Generators Using Keras

This project describes two different approaches and implimentations to making an LSTM text generater using the Python Library Keras. The first approach used a model that generated one character at a time (the char-based model) and another model that generated one word at a time (the word-based model). For context, these models were intended to be hosted on a Javascript webserver which could be queried as an API and provide generated text. 

The first approach used a char-based model that generated one character at a time. This character would then be appended to the end of the seed text and fed back into the model to predict the next character. This process was very time consuming. The code for building these models and instructions for how to use them can be found in the folder titled "char-based".

In an attempt to speed up the text generation, the second approach used a word-based model that generated full words at a time. This model worked very similarly to the one above. It would generate a word, append it to the seed text, and  attempt to predict the next word. Whenever the model encounters a word it has not seen before, it seems to skip over that word. This means that it may be necessary to add some level of buffer before the provided seed text so that the model will get seed text of the expected length or greater.

## The Char-Based Text Generator

This char-based model is based on the Keras example project given [here](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py).

For more detail, see the python notebook in the char-based folder.

## The Word-Based Text Generator

The word-based is model is based on the work of Jason Brownlee given [here](https://machinelearningmastery.com/how-to-develop-a-word-level-neural-language-model-in-keras/).

For more detail, see the python notebook in the word-based folder here. 

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
