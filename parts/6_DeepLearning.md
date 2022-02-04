# Deep Learning / Neural Networks

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

## Multi-Layer Perceptrons

Construction:
`Input layer -> Mapping (learned) -> Hidden layer -> Output layer`

Output:

$y_k(x) = g^{(2)} ( \sum_{i=0}^h W_{ki}^{(2)} g^{(1)} ( \sum_{j=0}^d W_{ij}^{(1)} x_j ) ) )$

With activation functions $g^{(k)}$.

E.g., $g^{(1)}(a) = a$, $g^{(2)}(a) = \sigma(a)$

_The hidden layer can have an arbitrary number of nodes_

Universal approximators
~ A 2-layer network (1 hidden layer) can approximate any continuous function of a compact domain arbitrarily well!

### Learning with Hidden Units

Networks without hidden units are very limited in what they can learn

- More layers of linear units do not help = still linear
- Fixed output non-linearities are not enough.

We need multiple layers of _adaptive_ non-linear hidden units.
But how can we train such nets?

- Need an efficient way of adapting all weights, not just the last layer.
- Learning the weights to the hidden units = learning features
- This is difficult, because nobody tells us what the hidden units should do.

-> Main challenge in deep learning.

#### Gradient Descent\

Two main steps:

1. Computing the gradients for each weight
2. Adjusting the weights in the direction of the gradient

Set up Error function $E(W) = \sum_n L(t_n; y(x_n;W)) + \lambda \Omega(W)$
with a loss $L$ and regularizer $\Omega$.

Then update each weight $W_{ij}^{(k)}$ in the direction of the gradient:
$\frac{\partial E(W)}{\partial W_{ij}^{(k)}}$

## Obtaining the Gradients

### Approach 1: Naive Analytical Differentiation

Compute the gradients for each variable analytically.

Multi-dimensional case: Total derivative
-> Need to sum over all paths that lead to the target variable $x$.

_What is the problem when doing this?_

- With increasing depth, there will be exponentially many paths!
- Infeasible to compute this way.

### Approach 2: Numerical Differentiation

Given the current state $W^{(\tau)}$, we can evaluate $E(W^{(\tau)})$.

_Idea:_ Make small changes to $W^{(\tau)}$ and accept those that improve $E(W^{(\tau))}$.

=> Horribly inefficient!
Need several forward passes for each weight.
Each forward pass is one run over the entire dataset!

### Approach 3: Incremental Analytical Differentiation (Backpropagation)

_Idea:_

- Compute the gradients layer by layer.
- Each layer below builds upon the results of the layer above.

The gradient is propagated backwards through the layers.

#### Backpropagation Algorithm\

1. Convert the discrepancy between each output and its target value into an error derivate.
2. Compute error derivatives in each hidden layer from error derivatives in the layer above.
3. Use error derivatives w.r.t. activities to get error derivatives w.r.t. the incoming weights.

Notation:

- $y_j^{(k)}$ Output of layer $k$ Connections: $z_j^{(k)} = \sum_i w_{ij}^{(k-1)} y_i^{(k-1)}$

- $z_j^{(k)}$ Input of layer $k$ $y_j^{(k)} = g(z_j^{(k)})$

Efficient propagation scheme:

- $y_i^{(k-1)}$ is already known from forward pass! (Dynamic Programming)

-> Propagate back the gradient from layer $k$ and multiply with $y_i^{(k-1)}$

```
Here the algorithms in psudocode
```

#### Analysis: Backpropagation\

Backpropagation is the key to make deep NNs tractable.

However the Backprop algorithm given here is specific to MLPs.

- It does not work with more complex architectures, e.g. skip connections or recurrent networks!
- Whenever a new connection function induces a different functional form of the chain rule,
  you have to derive a new Backprop algorithm for it.

## Learning Multi-layer Networks

### Computational Graphs

We can think of mathematical expressions as graphs.
I.e. we can divide every equation into a set of simple sub-equations.

From:
$e = (a+b) * (b+1)$

To:

$c= a + b$
$d = b + 1$
$e = c * d$

Which makes it easier to take derivates of those equations.

_Problem:_ Combinatorial explosion

_Solution:_

Efficient algorithms for computing the sum.

Instead of summing over all of the paths explicitly,
compute the sum more efficiently by merging paths back together at every node.

### Approach 4: Automatic Differentiation

- Convert the network into a computational graph.
- Each new layer/module just needs to specify how it affects the forward and backward passes.
- Apply reverse-mode differentiation.

=> Very general algorithm, used in today's Deep Learning packages.

#### Modular Implementation\

Solution in many current Deep Learning libraries

- Provide a limited form of automatic differentiation
- Restricted to "programs" composed of "modules" with predefined set of operations.

Module is defined by two main functions:

1. $y = module.fprop(x)$
   computes outputs $y$ given inputs $x$, where $x,y$ and intermediate results stored in the module

2. $\frac{\partial E}{\partial x} = module.bprop(\frac{\partial E}{\partial y})$
   computing the gradient $\partial E/ \partial x$ of a scalar cost w.r.t. the inputs $x$ given the gradient $\partial E/\partial y$ w.r.t. the outputs $y$

---

```
Here some stuff about Implementation Problems.
```

## Gradient Descent

Two main steps:

1. Computing the gradients for each weight
2. Adjusting the weights in the direction of the gradient

$w_{kj}^{(\tau + 1)} = w_{kj}^{(\tau)} - \eta \frac{\partial E(w)}{\partial w_{kj}} \arrowvert_{w^{(\tau)}}$

### Stochastic vs. Batch Learning

#### Batch learning\

Process the full dataset at once to compute the gradient.

$w_{kj}^{(\tau + 1)} = w_{kj}^{(\tau)} - \eta \frac{\partial E(w)}{\partial w_{kj}} \arrowvert_{w^{(\tau)}}$

Advantages:

- Conditions of convergence are well understood.
- Many acceleration techniques (e.g., conjugate gradients) only operate in batch learning.
- Theoretical analysis of the weight dynamics and convergence rates are simpler.

#### Stochastic learning\

- Choose a single example from the training set.
- Compute the gradient only based on this example
- This estimate will generally be noisy, which has some advantages.

$w_{kj}^{(\tau + 1)} = w_{kj}^{(\tau)} - \eta \frac{\partial E_{\color{red}n}(w)}{\partial w_{kj}} \arrowvert_{w^{(\tau)}}$

Advantages:

- Usually much faster than batch learning.
- Often results in better solutions.
- Can be used for tracking changes.

#### Minibatches\

Middle ground between batch and stochastic learning.

_Idea:_

- Process only a small batch of training examples together
- Start with a small batch size & increase it as training proceeds.

Advantages

- Gradients will be more stable than for stochastic gradient descent,
  but still faster to compute than with batch learning.
- Take advantage of redundancies in the training set.
- Matrix operations are more efficient than vector operations.

Caveat

Error function should be normalized by the minibatch size,
s.t. we can keep the same learning rate between minibatches

$E(W) = \frac{1}{N} \sum_{n} L(t_n, y(x_n;W)) + \frac{\lambda}{N} \Omega(W)$

### Choosing the Right Learning Rate

Considering a simple 1D example.
If E is quadratic, the optimal learning rate is given by the inverse of the Hessian.

_What happens if we exceed this learning rate?_

Instaed of getting closer to minimum, we start moving away from it fast.

### Momentum

**Batch Learning**

- Simplest case: steepest decent on the error surface.
- Updates perpendicular to contour lines

**Stochastic Learning**

- Simplest case: zig-zag around the direction of steepest descent.
- Updates perpendicular to constraints from training examples.

---

If the inputs are correlated, the ellipse will be elongated
and direction of steepest descent is almost perpendicular to the direction towards the minimum!

#### The Momentum Method\

_Idea_

Instead of using the gradient to change the position of the weight "particle", use it to change the velocity.

_Intuition_

- Example: Ball rolling on the error surface
- It starts off by following the error surface, but once it has accumulated momentum, it no longer does steepest decent.

_Effect_

- Dampen oscillations in directions of high curvature by combining gradients with opposite signs.
- Build up speed in directions with a gentle but consistent gradient.

_Behavior_

- If the error surface is a tilted plane, the ball reaches a terminal velocity

  - If the momentum $\alpha$ is close to 1, this is much faster than simple gradient descent.

- At the beginning of learning, there may be very large gradients.
  - Use a small momentum initially (e.g., $\alpha = 0.5$).
  - Once the large gradients have disappeared and the weights are stuck in a ravine,
    the momentum can be smoothly raised to its final value (e.g., $\alpha = 0.90$ or even $\alpha = 0.99$).
  - This allows us to learn at a rate that would cause divergent oscillations without the momentum.

#### Separate, Adaptive Learning Rates\

**Problem**

- In multilayer nets, the appropriate learning rates can vary widely between weights.
- The magnitudes of the gradients are often very different for the different layers,
  especially if the initial weights are small.
  - Gradients can get very small in the early layers of deep nets.
- The fan-in of a unit determines the size of the "overshoot" effect when changing multiple weights simultaneously to correct the same error.
  - The fan-in often varies widely between layers

**Solution**

- Use a global learning rate, multiplied by a local gain per weight (determined empirically)

### Better Adaptation: RMSProp

_Motivation_

- The magnitude of the gradient can be very different for different weights and can change during learning.
- This makes it hard to choose a single global learning rate.
- For batch learning, we can deal with this by only using the sign of the gradient,
  but we need to generalize this for minibatches.

_Idea_

Divide the gradient by a running average of its recent magnitude

$MeanSq (w_{ij}, t) = 0.9 MeanSq(w_{ij}, t-1)  + 0.1 (\frac{\partial E}{\partial w_{ij}} (t))^2$

Divide the gradient by $sqrt(MeanSq(w_{ij}, t))$.

#### Reducing the Learning Rate\

Final improvement step after convergence is reached

- Reduce learning rate by a factor of 10.
- Continue training for a few epochs.
- Do this 1-3 times, then stop training.

_Effect_

Turning down the learning rate will reduce the random fluctuations in the error
due to different gradients on different minibatches.

### Summary

Deep multi-layer networks are very powerful.

_But training them is hard!_

- Complex, non-convex learning problem
- Local optimization with stochastic gradient descent

_Main issue:_ getting good gradient updates for the early layers of the network

- Many seemingly small details matter!
- Weight initialization, normalization, data augmentation, choice of nonlinearities, choice of learning rate, choice of optimizer,...
