# Sequence to Sequence Models

## Introduction 
Sequence-to-Sequence (Seq2Seq) models are one of the most important classes of deep learning models, with a wide range of applications, including text modeling, time series analysis, audio processing, and many others. Most Seq2Seq models are based on the seminal paper titled "Sequence to Sequence Learning with Neural Networks" by Ilya Sutskever, Oriol Vinyals, and Quoc V. Le (Google Team) in 2014 [https://arxiv.org/abs/1409.3215]. Since then, many important and useful neural network architectures have been proposed. Before delving into those, let's first understand why sequence-to-sequence modeling is so significant.

If we examine common datasets, they often fall into one of the following categories:

Uncorrelated (Tabular) data: This is the most common type of data, where different data points are organized into rows, and columns represent the features of the data. This type of data doesn't rely on the ordering of data points. To model this kind of data, we can employ any architecture that doesn't consider data ordering. For example, when fitting a line to data (X, y), the modeling remains unchanged even if we interchange data points.

Correlated data: This type of data exhibits correlations between neighboring data points, either in space, time, or both. For example, in images, we know that each pixel is correlated with its neighboring pixels, which is what makes an image unique. Imagine randomizing the pixels, and you'd lose the image's structure. Similar correlations exist in time series data, where future values depend on past values, often spanning multiple time steps. In the context of language, when reading from left to right, it's evident that future words depend on the current and preceding words, creating an ordered sequence. Ordered data sequences like these can be found in various domains, which is the main focus of this project.

## How to Model a Sequence

There are multiple ways to model a sequence, but they all must incorporate a memory of the past, a feature provided by Recurrent Neural Networks (RNNs). Before delving into that, let's revisit the fundamental problem in machine learning: the input-output mapping.

Imagine we have an input 'X' and an output 'y.' The task of a machine learning model is to find a mapping, typically nonlinear, that transforms X into y:

y = f(X)

Consider the common case of the 'sigmoid' nonlinearity:

y = sigmoid(W * X)

Here, 'W' represents the fitting parameters or 'weights,' and 'b' is the bias term. In practice, we can incorporate the bias 'b' into 'W' by adding a constant to the data 'X,' resulting in:

y = sigmoid(W * X)

We can replace the sigmoid function with any other nonlinear function 'f.'

When we apply this function to a series of data points (X1, X2, X3, ...), we obtain corresponding outputs (y1, y2, y3, ...). In this context, the order of data points doesn't matter, as 'W' doesn't possess memory.

To introduce memory into the model, we need to modify the equation as follows:

h(t) = f(W * X(t) + U * h(t-1))
y(t) = g(h(t))

Here, 'h' represents the memory state, and 'g' is another nonlinear function.

Expanding on the equation:

h(t) = f(X(t) + U * (f(W * X(t-1) + U * h(t-2))))

During the machine learning training phase, when we use the above formula, we update the values of the weight matrices 'W' and 'U.' This is how the training process occurs.

## Back Propagation 

Unline in other neural networks, we use back propagation for RNN networks also but there is some problem. Let us look at how weights are update in neural networks
under the back propagation scheme. 

In Supervised Machine Leraning we try to minimize a scoring function called the loss function which measures the mismatch between the actual outputs and the predicted output from the neural networks. We keep 'adjusting' the weights of the networks till the mismatch stops decreasing. 

How much we need to adjust a weight depend on by how much amount it is bringing down the loss for a unit change in the weight ie., dL/dw.
Now the loss function is directly related to the wights of the last layer (output layer) but indirectly related to the weights of the earlier layers.


The loss is a function of actual output y(n) and predicted output yhat(n) where yhat(n) is the function of weights of the last layer.
If the outputs of the layers from the left are yhat(1), yhat(2), ...yhat(n) then

dl/dw(n-2) = dl / dyhat (n) * dyhat (n)/dyhat (n-1) * dyhat (n-1) / dyhat (n-2) * dyhat (n-2) /  dw(n-2)

The gradiant at the left is the product of many grdiants at left which can be very small and can make the left hand side very small - this is called vanishing 
gradiant problem. 





## Tuturials 

- [Encoder-Decoder Seq2Seq Models, Clearly Explained!!](https://medium.com/analytics-vidhya/encoder-decoder-seq2seq-models-clearly-explained-c34186fbf49b)
- [Seq2Seq Models : French to English translation using encoder-decoder model with attention.](https://medium.com/analytics-vidhya/seq2seq-models-french-to-english-translation-using-encoder-decoder-model-with-attention-9c05b2c09af8) 

- [A ten-minute introduction to sequence-to-sequence learning in Keras](https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html)
- [Understanding LSTM Networks ](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)


## References 

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Code-- Ishika]






