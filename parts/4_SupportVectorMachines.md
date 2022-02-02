# Support Vector Machines

## Softmax Regression

### Multi-class generalization of logistic regression

Assume binary labels $t_n \in \{0, 1\}$.

Softmax generalizes thisto $K$ values in 1-of-$K$ notation.

\begin{equation}
y(x;w) =
\begin{bmatrix}
P(y = 1 | x;u) \\
P(y = 2 | x;w) \\
\vdots \\
P(y = k | x;W)
\end{bmatrix}
\end{equation}

With _softmax_ function: $\frac{exp(a_k)}{\sum_j exp(a_j)}$

#### Logistic Regression\

Alternative way of writing the cost function.

$E(w) = \sum_{n=1}^N \sum_{k=0}^1 \{ \mathbb{I}(t_n = k) \} \ln P(y_n = k | x_n;w) \}$

#### Softmax Regression\

Generalization to K classes using indicator functions.

$E(w) = \sum_{n=1}^N \sum_{k=0}^1 \{ \mathbb{I}(t_n = k) \} \ln \frac{exp(w_k^T x)}{\sum_{j=1}^K exp(w_j^T x)}  \}$

### Optimization

Again, no closed-form solution is available — Resort again to Gradient Descent.

$\nabla_{w_k} E(w) = - \sum^N_{n=1} [ \mathbb{I}(t_n=k) \ln P(y_n=k | x_n;w) ]$

We can now plug this into a standard optimization package.

## Support Vector Machines

### Motivation

Goal: predict class labels of new observations.

But: as training progresses we start to overfit to training data.

Popular solution: Cross-validation.

- Split the available data into training and validation sets.
- Estimate the generalization error based on the error on the validation set.
- Choose the model with minimal validation error.

With linearly separable data there are many ways to split the data.
The problem is how to choose the best split?

Intuitively, we would like to select the classifier which leaves maximal "safety room" for future data points.

This can be obtained by maximizing the margin between positive and negative data points.

=> The SVM formulates this problem as a convex optimization problem. I.e., we can find optimal solution.

### SVM

Consider linearly separable data:

- $N$ training points $\{(x_i, y_i)\}_{i=1}^N$ $x_i \in \mathbb{R}^d$
- Target values $t_i \in \{-1, 1\}$
- Hyperplane separating the two classes: $w^T x + b = 0$

Then Canonical representation of the decision hyperplane:

$t_n(w^T x_n + b) \geq 1 \forall n$

**Optimization problem**

Find the hyperplane satisfying: $arg_{w,b} min \frac{1}{2} \| w \|^2$

under constraints: $t_n(w^T x_n + b) \geq 1 \forall n$

- Quadratic programming problem with linear constraints.
- Can be formulated using Lagrange multipliers (see slides).

### Lagrangian Formulation

$L_p = \frac{1}{2} \|w\|^2 - \sum_{n=1}^N a_n \{t_ny(x_n)-1\}$

Under conditions:

- $a_n \geq 0$
- $t_n y(x_n) -1 \geq 0$
- $a_n \{t_ny(x_n)-1\} = 0$

### Solution

Computed as a linear combination of the training examples: $w = \sum_{n=1}^N a_n t_n x_n$.

Because of the KKT conditions, the following must also hold: $a_n(t_n (w^T x_n + b) - 1) = 0$.

This implies that $a_n > 0$ only for training data points for which $t_n(w^T x_n + b) - 1 = 0$.

=> _Only some of the data points actually influence the decision boundary!_

To define the decision boundary, we still need to know b:

$b = \frac{1}{N_\mathcal{S}}  \sum_{n \in \mathcal{S}} (t_n - \sum_{m \in \mathcal{S}} a_m t_m x_m^T x_n)$

### Discussion

Linear SVM

- Linear classifier
- SVMs have a "guaranteed" generalization capability.
- Formulation as convex optimization problem.

=> Globally optimal solution!

Primal form formulation

- Solution to quadratic prog. problem in $M$ variables is in $\mathcal{O}(M^3)$.
- Here: $D$ variables -> $\mathcal{O}(D^3)$.
- Problem: scaling with high-dim. data ("curse of dimensionality")

### Dual form formulation

```
Does something that I don't understand.
```

At the end we should maximize:

$L_d (a) = \sum_{n=1}^N a_n - \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_n a_m t_n t_m (x_m^T x_n)$

Under conditions: $a_n \geq 0 \forall n$ and $\sum_{n=1}^N a_n t_n = 0$.

Then the hyperplane is given by the $N_\mathcal{S}$ support vectors:

$w = \sum_{n = 1}^\mathcal{S} a_n t_n x_n$

### Soft-margin classification

Solution above works only if data linearly separable.
But we can add some tolerance to this division to make it work even if outliers present.

We add "_slack variable_" $\xi$ to the formulation:

$w^T x_n + b \geq 1 - \xi$ for $t_n = 1$
$w^T x_n + b * -1 + \xi$ for $t_n = -1$

where $\xi_n \geq 0 \forall n$.

**Interpertation**

- $\xi_n = 0$ : point on correct side of the margin
- $\xi_n = | t_n - y(x_n) |$ : otherwise
  - $\xi_n > 1$ : misclassified point

> Note:
>
> We do not have to set the slack variables ourselves!
> They are jointly optimized together with $w$.

**New** Dual Formulation

$L_d (a) = \sum_{n=1}^N a_n - \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_n a_m t_n t_m (x_m^T x_n)$

Under conditions: $a_n ? 0 ? C$ and $\sum_{n=1}^N a_n t_n = 0$.

## Nonlinear Support Vector Machines

Not everything can be linearly separated, so we need nonlinear classifiers.

**General idea:**

The original input space can be mapped to some higher-dimensional feature space
where the training set is separable.

Nonlinear transformation $\phi$ of the data points $x_n$:

$x \in \mathbb{R}^D$ $\phi : \mathbb{R}^D \to \mathcal{H}$

Hyperplane in higher dimensional space $\mathcal{H}$:

$w^T \phi(x) + b = 0$

### Problem with High-dim. Basis Functions

In order to apply the SVM, we need to evaluate the function: $y(x) = w^T \phi(x) + b$.

Using hyperplane $w = \sum_{n=1}^N a_n t_n \phi(x_n)$.

Which leads to Problems in high dimensional feature space.

**Solution:**

We can replace dot product $\phi(x)^T \phi(x)$ by a kernel function: $k(x,y)$.

Then $y(x) = \sum_{n=1}^N a_n t_n k(x_n, x) + b$.

The kernel function implicitly maps the data to the higher dimensional space (without having to compute $\phi(x)$ explicitly)!

_But it only works fore specific kernel functions._

---

**"Every positive definite symmetric function is a kernel."**
~ Mercer's theorem (modernized version)

---

(positive definite = all eigenvalues are > 0)

Example kernels:

- Polynomial kernel: $k(x,y) = (x^T y + 1)^p$
- Radial Basis Function kernel (e.g. Gaussian): $k(x,y) = e^{-\frac{||x-y||^2}{2\sigma^2}}$

**New New** Dual Formulation

$L_d (a) = \sum_{n=1}^N a_n - \frac{1}{2} \sum_{n=1}^N \sum_{m=1}^N a_n a_m t_n t_m k(x_m,x_n)$

Under conditions: $a_n ? 0 ? C$ and $\sum_{n=1}^N a_n t_n = 0$.

## Summary

### Properties

- in practice work very well
- among the best performers for a number of classification tasks ranging from text to genomic data
- can be applied to complex data types by designing kernel functions for such data
- The kernel trick has been used for a wide variety of applications. It can be applied wherever dot products are in use

### Limitaitons

- How to select the right kernel?
  - Best practice guidelines are available for many applications
- How to select the kernel parameters?
  - (Massive) cross-validation.
  - Usually, several parameters are optimized together in a grid search.
- Solving the quadratic programming problem
  - Standard QP solvers do not perform too well on SVM task.
  - Dedicated methods have been developed for this, e.g. SMO.
- Speed of evaluation
  - Evaluating $y(x)$ scales linearly in the number of SVs.
  - Too expensive if we have a large number of support vectors.
    -> There are techniques to reduce the effective SV set.
- Training for very large datasets (millions of data points)
  - Stochastic gradient descent and other approximations can be used

### Error Function

SVMs result in so-called _hinge error_.
Where the error is minimized and correct classification is constant.

This leads to:

- sparsity - Zero error for points outside the margin
- robustness - Linear penalty for misclassified points

## Applications

### Text Classification

**Problem:** Classify a document in a number of categories.

Representation:

- "Bag-of-words" approach
- Histogram of word counts (on learned dictionary)

Usage:

- spam filters
- ocr - optical character recognition
- object detection
