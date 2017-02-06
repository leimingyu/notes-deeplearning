## Overfitting
The strategy here is to find hyperparameters such that the error on the training set is low, but you're not overfitting to the data. If you train the network too long or have too many hidden nodes, it can become overly specific to the training set and will fail to generalize to the validation set. That is, the loss on the validation set will start increasing as the training set loss drops.

This is the number of times the dataset will pass through the network, each time updating the weights. As the number of **epochs** increases, the network becomes better and better at predicting the targets in the training set. You'll need to choose enough epochs to train the network well but not too many or you'll be overfitting.



## Learning resources:
* https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/
* https://www.oreilly.com/learning/hello-tensorflow
* https://www.tensorflow.org/tutorials/mnist/beginners/
* https://github.com/aymericdamien/TensorFlow-Examples
* https://www.youtube.com/watch?v=2FmcHiLCwTU&t=84s
* https://cs224d.stanford.edu/lectures/CS224d-Lecture7.pdf
