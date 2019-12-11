# Movie Sentiment Analysis

## Dataset
Kaggle: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
* Records: 50,000
* Columns: 2
    * review
    * sentiment - "positive" and "negative" => binary classification problem

## Dependencies
* Pandas
* Seaborn
* Numpy
* Scikit-learn
* Tensorflow
* Keras
* Matplotlib
* Pickle

`pip install -r requirements.txt`

## Deep Learning using Neural Networks

### Simple Neural Network
* Sequential model
* One Embedding layer
* Flattening layer
* Dense layer
    * activation function
<br>
[Notebook]('model_NN.ipynb')

### Convolutional Neural Network (CNN)
Primarily used for 2D data classification, such as images. Work well with 1D text data as well. Tries to find specific features in the first layer. In the next layers, the initially detected features are joined together to form bigger features. Ref: https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
<br>
* Sequential model
* One Embedding layer
* 1D convolutional layer
    * features or kernels
    * activation function
* Global max pooling layer
    * reduce feature size
* Dense layer
    * activation function
<br>
[Notebook]('model_CNN.ipynb')

### Recurrent Neural Network (CNN)
**Long Short Term Memory Network** (LSTM)<br>
Recurrent neural networks variant
<br>
* Sequential model
* One Embedding layer
* LSTM layer
    * neurons
* Dense layer
    * activation function
<br>
[Notebook]('model_RNN.ipynb')

## Techniques used
* ***Keras*** **Embedding Layer**
* ***Stanford CoreNLP*** **GloVe** word embeddings

## Conclusion
* The *difference between the accuracy values* for training and test sets is much smaller in **Recurrent NN** as compared to that in **Simple NN** and **Convolutional NN**.
* The *difference between the loss values is negligible* in Recurrent NN.
    * Model is NOT overfitting
<br>
So RNN is the best best algorithm for the model for our text classification.

## Considerations
The number of layers, neurons, hyper parameters values, activation functions etc. can be changed to find the best NN model.