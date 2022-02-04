# Optimization

## Tricks of the Trade

### Shuffling the Examples

_Idea:_
Network learns faster from most unexpected sample.
It is advisable to choose a sample at each iteration that is most
unfamiliar to the system.

I.e., **do not present all samples of class A, then all of class B**.

A large relative error indicates that an input has not been learned by the network yet,
so it contains a lot of information.
It can make sense to present such inputs more frequently.
**But**: be careful, this can be disastrous when the data are outliers.

When working with stochastic gradient descent or minibatches, make use of shuffling.

### Data augmentation

_Idea:_
Augment original data with synthetic variations to reduce overfitting.

Examples of augmentation:

- Cropping
- Zooming
- Flipping
- Color PCA (color space transformation)

_Effect:_

- Much larger training set
- Robustness against expected variations

### Normalization

_Motivation:_

Consider the Gradient Descent update steps.
When all of the components of the input vector $y_i$ are positive,
all of the updates of weights that feed into a node will be of the same sign.

Leads to:

- Weights can only all increase or decrease together.
- Slow convergence

Convergence is the fastest if:

- The mean of each input variable over the training set is zero.
- The inputs are scaled such that all have the same covariance.
- Input variables are uncorrelated if possible.

## Nonlinearities

|                    |                                                    |
| ------------------ | -------------------------------------------------- |
| Sigmoid            | $g(a) = \sigma(a) = \frac{1}{1 + exp\{-a\} }$      |
| Hyperbolic tangent | $g(a) = \tanh(a) = 2 \sigma (2a) -1$               |
| Softmax            | $g(a) = \frac{exp \{-a_i\} }{\sum_j exp \{-a_j\}}$ |

---

Normalization is also important for intermediate layers:
Symmetric sigmoids, such as tanh, often converge faster than the standard logistic sigmoid.

Recommended sigmoid:

$f(x) = 1.7159 tanh (\frac{2}{3} x)$

When used with transformed inputs, the variance of the outputs will be close to 1.

### Usage

_Output nodes_ - Typically, a sigmoid or tanh function is used here:

- Sigmoid for nice probabilistic interpretation (range [0,1]).
- tanh for regression tasks

_Internal nodes_:

- Historically, tanh was most often used.
- tanh is better than sigmoid for internal nodes, since it is already centered.
- Internally, tanh is often implemented as piecewise linear function (similar to hard tanh and maxout).
- More recently: ReLU often used for classification tasks.

Effects of sigmoid/tanh function:

- Linear behavior around 0
- Saturation for large inputs

### Extension: ReLU

Improvement for learning deep models.
That uses Rectified Linear Units (ReLU) instead of sigmoid: $g(a) =max\{0, a\}$.

Effect gradient is propagated with a constant factor:
$\frac{\partial g(a)}{\partial a} = \left\{ \begin{array}{ll} 1 & \mbox{if } a > 0 \\ 0 & \mbox{otherwise} \end{array} \right.$

_Advantages_:

- Much easier to propagate gradients through deep networks.
- We do not need to store the ReLU output separately
  - Reduction of the required memory by half compared to tanh!

_Disadvantages / Limitations_:

- A certain fraction of units will remain "stuck at zero".
  - If the initial weights are chosen such that the ReLU output is 0 for the entire training set,
    the unit will never pass through a gradient to change those weights.
- ReLU has an _offset bias_, since its outputs will always be positive

#### Further Extensions\

|            |                                                                                         |
| ---------- | --------------------------------------------------------------------------------------- |
| ReLU       | $g(a) = max\{0, a\}$                                                                    |
|            |                                                                                         |
| Leaky ReLU | $g(a) = max\{\beta a, a\}$                                                              |
|            | Avoids stuck-at-zero units                                                              |
|            | Weaker offset bias                                                                      |
|            |                                                                                         |
| ELU        | $g(a) =  \left\{ \begin{array}{ll} a & x < 0 \\ e^a - 1 & x \geq 0 \end{array} \right.$ |
|            | No offset bias anymore                                                                  |
|            | BUT: need to store activations                                                          |

## Initialization

### Initializing the Weights

_Motivation_:

- The starting values of the weights can have a significant effect on the training process.
- Weights should be chosen randomly, but in a way that the sigmoid is primarily activated in its linear region.

Assuming that:

- The training set has been normalized
- The recommended sigmoid $f(x) = 1.7159 tanh(\frac{2}{3})$ is used

the initial weights should be randomly drawn from a distribution (e.g., uniform or Normal) with mean zero and variance:

$\sigma_w^2 = \frac{1}{n_{in}}$

where $n_{in}$ is the fan-in (#connections into the node).

### Glorot Initialization

In 2010, Xavier Glorot published an analysis of what went wrong in
the initialization and derived a more general method for automatic
initialization.

This new initialization massively improved results and made direct
learning of deep networks possible overnight.

#### Analysis\

```
Some maths
```

## Advanced techniques

### Batch Normalization

_Motivation_:
Optimization works best if all inputs of a layer are normalized.

_Idea_

- Introduce intermediate layer that centers the activations of the previous layer per minibatch.
- I.e., perform transformations on all activations and undo those transformations when backpropagating gradients
- **Complication**: centering + normalization also needs to be done at test time,
  but minibatches are no longer available at that point.
  - Learn the normalization parameters to compensate for the expected bias of the previous layer (usually a simple moving average)

_Effect_

- Much improved convergence (but parameter values are important!)
- Widely used in practice

### Dropout

_Idea_:

- Randomly switch off units during training (a form of regularization).
- Change network architecture for each minibatch,
  effectively training many different variants of the network.
- When applying the trained network,
  multiply activations with the probability that the unit was set to zero during training.

=> Greatly improved performance
