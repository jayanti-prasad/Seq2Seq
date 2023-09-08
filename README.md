#Sequence to Sequence Models

Sequence to Sequence (Seq2Seq) models are one of the most important class of deep learning models
which have a wide range of applications - from modeling text, time series to audio and many others.
Most of the Seq2Seq models are based on the work of a paper titled "Sequence to Sequence Learning with Neural Networks" 
by [https://arxiv.org/abs/1409.3215] (Ilya Sutskever, Oriol Vinyals, Quoc V. Le (Google Team)) in 2014. 
Since then many important and useful neural network architecrurs have been proposed. Before going to 
discuss about those let us first understand why sequence to sequence  modeling is so important. 

If we look at common data sets they may fall in one of the following catagories:

- Tabular data : This is the most common type of data in wich different data points are placed into different rows and
the columns represent the feature of the data. This type of data has the propery that the ordering of the data points is
no important. In order to model this kind of data we can employ any architecture which does not respect the ordering. For example when we fit a line to data (X, y) the modeling does not chance even we interchange data points.


- Correlated data : This is kind of data in which the neighbouring data points are correlated. This neighbouhood can be in 
space or time or in the both. For example, in case of an image we know that any pixel is correlated to its neighbouring pixels,
in fact this is what makes an image unique. This about randomizing the pixel and you will lose the image. These correlation exist in time series also - what will happen tomorrow depen of what happes today or even many days before today - in simple words past predicts future. In case of time series we have one way correlations. If we look at normal english language text where we 
read from left to right and it is easy to understand that what word will come into the future depends on what word we have at
present and before that also. It is a kind of ordered sequence. We may find these sequences (orderd data points) in many domian and that is focus topic of this project here. 

## How to model a sequence 
There are multiple ways to model a sequence but they all must share a feature - they must have some memory of the past and this
is what Recurrent Neural Networks or RNN provides. Before going to dicuss that let us recall the fundamental problem of the machine learning - Input - Output mapping.

Let us consider we have input 'X' and output 'y' and so the task of a machine learning model is to find a mapping (non-linear 
in most cases) that maps X to y:

y = f (X)

Let us consider the most common case of 'sigmoid' non-linearity 


y = sigmoid (W *X + b )

Here W, are fitting parameters or 'weights' and 'b' is called the bias. In fact we can add one constant in data 'X' and absorb 
'b' into 'W' and write :

y = sigmoid (W * X)

In place of sigmoid, we can use any nonliear function 'f'


If we start passing data points (X1, X2, X3, ...) etc to the above function we will get (y1, y2, y3,...) where ordering 
will not matter - since W does not have a memory.

In order to have a memory we need to modify the above equation in the following way:

h(t) = f (W X(t) + U * h (t-1)) 
y(t) = g ( h(t))

Here 'h' is called the mory state and 'g' is another  non-liner function. 


Let us expand the above equation :

h(t) = f ( X(t) + U ( f W(X(t-1) + U * h(t-2)))



























https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html
