# Deep Learning

## A Brief History of Neural Networks

**1957 Rosenblatt invents the Perceptron**

- And a cool learning algorithm: "Perceptron Learning"
- Hardware implementation "Mark | Perceptron" for 20x20 pixel image analysis

**1969 Minsky & Papert**

- Showed that (single-layer) Perceptrons cannot solve all problems.
- This was misunderstood by many that they were worthless.

**1980s Resurgence of Neural Networks**

- Some notable successes with multi layer perceptrons.
- Backpropagation learning algorithm
- But they are hard to train, tend to overfit, and have unintuitive parameters.
- So, the excitement fades again...

**1995+ Interest shifts to other learning methods**

- Notably Support Vector Machines
- Machine Learning becomes a discipline of its own.
- The general public and the press still love Neural Networks.

**2005+ Gradual progress**

- Better understanding how to successfully train deep networks
- Availability of large datasets and powerful GPUs
- Still largely under the radar for many disciplines applying ML

**2012 Breakthrough results**

- ImageNet Large Scale Visual Recognition Challenge
- A ConvNet halves the error rate of dedicated vision approaches.
- Deep Learning is widely adopted.

## Perceptrons

### Standard Perceptron

Construction: `Input layer -> Weights -> Output layer`

Input Layer := Hand-designed features based on common sense

Outputs:

- linear: $y(x) = w^Tx + w_0$
- logistic: $y(x) = \sigma(w^Tx + w_0)$

Learning := Determining the weights $w$

#### Multi-Class Networks\

Construction: `Input layer -> Weights -> Output layer`

_One output node per class_

Outputs:

- linear: $y_k(x) = \sum_{i=0}^d W_{ki} x_i$
- logistic: $y_k(x) = \sigma ( \sum_{i=0}^d W_{ki} x_i )$

Can be used to do multidimensional linear regression or multiclass classification.

#### Non-Linear Basis Functions\

Construction: `Input layer -> Mapping (fixed) -> Feature Layer -> Weights -> Output layer`

Outputs:

- linear: $y_k(x) = \sum_{i=0}^d W_{ki} \phi(x_i)$
- logistic: $y_k(x) = \sigma ( \sum_{i=0}^d W_{ki} \phi(x_i) )$

> Notes:
>
> - Perceptrons are generalized linear discriminants!
> - Everything we know about the latter can also be applied here.
> - feature functions $\phi(x)$ are kept fixed, not learned!

### Perceptron Learning

- Very simple algorithm

- Process the training cases in some permutation

  - If the output unit is correct, leave the weights alone.
  - If the output unit incorrectly outputs a zero, add the input vector to the weight vector.
  - If the output unit incorrectly outputs a one, subtract the input vector from the weight vector.

- This is guaranteed to converge to a correct solution if such a solution exists.

In this process we can use many loss functions:

- L2 loss
- L1 loss
- (Binary) cross-entropy loss
- Hinge loss
- Softmax cross-entropy loss

#### Regularization\

In addition, we can apply regularizers.

$E(w) = \sum_n L(t_n; y(x_n;w)) + \lambda \|w\|^2$

- This is known as weight decay in Neural Networks.
- We can also apply other regularizers, e.g. L1 => sparsity
- Since Neural Networks often have many parameters, regularization becomes very important in practice.

### Limitations of Perceptrons

_What makes the task difficult?_

Perceptrons with fixed, hand-coded input features can model any
separable function perfectly... given the right input features.

Once the hand-coded features have been determined,
there are very strong limitations on what a perceptron can learn.
-> Classic example: XOR function.

**A linear classifier cannot solve certain problems**

However, with a non-linear classifier based on
the right kind of features, the problem becomes solvable.
