# Neuronal network in Kotlin

Neuronal network is a statistical model where parameters can be updated with back propagation on a target. The implementation in this repository has been done with pure Kotlin, the backpropagation was done manually without any help of automatic differentiation framework like `Tensorflow`, `Pytorch` or `Deeplearning4j`. Following features are supported:

- [x] GUI application with Java Swing for interacting with MNIST dataset
- [x] Regression with MAE loss
- [x] Classification with sigmoid loss and cross entropy loss
- [x] Gradient descent optimization
- [x] Momentum driven gradient descent
- [ ] Nestorov driven gradient descent
- [x] Adagrad optimization
- [x] Adam optimization
- [ ] Dropping out neuronal network in forward and backward propagation

The implementation is made mostly for educational purpose to provide some insight into low level programming 
with neuronal network. For this purpose I implemented the whole network topology with Kotlin/Java.

### Dataset
- MNIST dataset for number classification
- Fashion-NIST dataset of Zalando Research for clothes classification
- Advanced Boston House Price dataset for price regression

### Dependencies
- `Apache Commons CSV` for CSV reading.
- `XChart` for visualization.
- `Nd4j` for linear algebra.

# Experiments with the implementation

### Performance evaluation for regression purpose

The performance validation of the neuronal network was made on the advanced [House Price Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).
For the training purpose, the data of the dataset has to be processed. Each discrete column will be one-hot-encoded and each continuous-valued column will be 
scaled to `[0, 1]`.

![](images/001-gradient-descent.png)

### Comparing gradient descent with momentum driven gradient descent

The same models are replicated, but the the training process is optimized by two different methods. The first one is the vanilla
gradient descent. The second one memorizes the velocity of past epochs and results into a better result. Both tested on the same data of the [House Price Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).

![](images/002-gradient-descent-and-momentum.png)

### Performance evaluation for classification

For classification purpose the output of the network will be scaled with a softmax function and the loss is calculated with the cross entropy 
between the ground truth and the output.

![](images/003-cross-entropy.png)
