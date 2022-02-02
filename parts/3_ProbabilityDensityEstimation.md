# Probability Density Estimation

How can we estimate (= learn) those probability densities?

In Supervised training case: data and class labels are known.
So we can estimate the probability density for each class separately.

## The Gaussian (or Normal) Distribution

One-dimensional

- Mean $\mu$
- Variance $\sigma^2$

$\mathcal{N}(x | \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi} \sigma} exp \{ - \frac{(x - \mu)^2}{2\sigma^2}\}$

Multi-dimensional

- Mean $\mu$
- Variance $\Sigma$

\begin{equation}
\mathcal{N}(x | \mu, \Sigma)
= \frac{1}{(2 \pi)^\frac{D}{2} |\Sigma|^\frac{1}{2}}
exp \{ -\frac{1}{2}(x - \mu)^{T} \Sigma^{-1}(x - \mu) \}
\end{equation}

### Properties

#### **Central Limit Theorem**

"The distribution of the sum of N i.i.d. random variables becomes increasingly Gaussian as N grows."

```
[There was some more stuff in slides but it was boring math]
```

_The marginals of a Gaussian are again Gaussians_

## Parametric Methods

Given some data $X$ with parameters $\theta = (\mu, \sigma)$.
What is the probability that $X$ has been generated from a
probability density with parameters $\theta$.

$L(\theta) = p(X|\theta)$

### Maximum Likelihood Approach

Single data point:

$p(x_n|\theta) = \frac{1}{\sqrt{2 \pi} \sigma} exp \{ - \frac{(x_n - \mu)^2}{2\sigma^2}\}$

Assumption all data points are independent:

$L(\\theta) = p(X|\theta) = \prod_{n=1}^N p(x_n|\theta)$

Log-Likelihood:
$E(\theta) = -ln L(\theta) = -\sum_{n=1}^N ln p(x_n|\theta)$

**Goal:**
Minimize $E(\theta)$.

How to:

1. Take the derivate of $E(\theta)$
2. Set it to zero
3. Quic Mafs

**Warning**

MLA is _biased_ - it underestimates the true variance.
I.e., _overfits to the observed data_.

#### Frequentist vs Bayesian

|               | Frequentist                                                            | Bayesian                                                |
| ------------- | ---------------------------------------------------------------------- | ------------------------------------------------------- |
| probabilities | are frequencies of random, repeatable events                           | quantify the uncertainty about certain states or events |
|               | fixed, but can be estimated more precisely when more data is available | uncertainty can be revised in the light of new evidence |

## Non-Parametric Methods

Often the functional form of the distribution is unknown.
So we have to estimate probability from data.

### Histograms

Partition data into bins with widths $\Delta_i$ and count number of observations in each bin.

$p_i = \frac{n_i}{N\Delta_i}$

_Often:_ $\Delta_i = \Delta_j \forall i,j$

| Advantages                              | Disadvantages           |
| --------------------------------------- | ----------------------- |
| works in any dimension $D$              | curse of dimensionality |
| no need to store data after computation | rather brute force      |

Bin size:

- too large: too much smoothing
- too small: too much noise

### Kernel Density Estimation

#### Parzen Window\

Idea: Hypercube of dimension $D$ with width edge length $h$.

We place a kernel window k at location x and count how many data points fall inside it.
Crude solution because the chosen function $k$ creates hard cuts around Hypercubes.

Kernel Function:
$k(u) = \begin{cases}1, |u_i * \frac{1}{2},i=1, ..., D\\0, else\end{cases}$

$K = \sum_{i=1}^N k(\frac{x-x_n}{h})$

$V = \int k(u) du = h^d$

Then probability density is:

$p(x) \simeq \frac{K}{NV} = \frac{1}{Nh_D} \sum_{i=1}^N k(\frac{x-x_n}{h})$

#### Gausian Kernel\

Similar to Parzen Window, but with Gaussian kernel function $k$.

$k(u) = \frac{1}{(2 \times \pi h^{2})^{1/2}} \exp \{ -\frac{u^{2}}{2h^{2}}\}$

$K = \sum_{i=1}^N k(x-x_n)$

$V = \int k(u) du = 1$

Then probability density is:

$p(x) \simeq \frac{K}{NV} = \frac{1}{N} \sum_{i=1}^N \frac{1}{(2 \pi h^{2})^{D/2}} \exp \{ -\frac{\| x - x_n \|^{2}}{2h^{2}}\}$

### K-Nearest Neighbors

Similar to above but we fix $K$ (the number of neighbors) and calculate $V$ (size of the neighbourhood).

Then: $p(x) \simeq \frac{K}{NV}$

_Warning:_
Strictly speaking, the model produced by K-NN is not a true density model,
because the integral over all space diverges.

#### Bayesian Classification\

$p(C_j | x) = \frac{p(x|C_j)p(C_j)}{p(x)}$

### Summary

- Very General
- Training requires no computation
- Requires storing and computing the entire dataset
  -> cost linear in the number of data points
  -> can be saved in implementation

Kernel size $K$ in K-NN?

- Too large: too much smoothing
- Too small: too much noise

#### Bias-Variance Tradeoff\

|           | Histograms                   | Kernel methods          | K-Nearest Neighbors     |
| --------- | ---------------------------- | ----------------------- | ----------------------- |
|           | _bin size? $\Delta$_         | _kernel size? $h$_      | _$K$?_                  |
| too large | $\Delta$ : too smoth         | $h$ : too smooth        | $K$ : too smooth        |
| too small | $\Delta$ : not smooth enough | $h$ : not smooth enough | $K$ : not smooth enough |

## Mixture Distribution

Motivation: single parametric distribution is often not sufficient, e.g., for multimodal data.

### Mixture of Gaussians

Idea: we take multiple Gaussians and combine them, giving each one a weight that also depends on input.

$p(x|\theta) = \sum_{j=1}^N p(x|\theta_j)p(j)$

Where:

- $p(x|\theta_j)$ is the probability of measurement $x$ given the j-th mixture component
- $p(j)$ is the prior of component $j$

> Note:
>
> $\int p(x)dx = 1$

### Maximum Likelihood

$E = - ln L(\theta) = - \sum_{n=1}^N ln p(x_n|\theta)$

$ln p(X|\pi, \mu, \Sigma) = \sum_{n=1}^N \{ \sum_{k=1}^K \pi_k \mathcal{N}(x_n|\mu_k, \Sigma_k) \}$

-> leads to infinite loop, because gausians depend on gausians
-> It is possible to apply iterative numerical optimization here,
but in the following, we will see a simpler method.

### K-Means Clustering

Iterative procedure:

1. Pick K random data points as initial cluster centres
2. Assign each data point to the closest cluster centre
3. Compute the new cluster centres as the mean of the assigned data points
4. Repeat until convergence

_Guaranteed to converge after finite number of iterations_

But:

- local optimum
- depends on the initial cluster centres

> Note:
>
> Can be used e.g. for image compression

| Advantages                 | Disadvantages                        |
| -------------------------- | ------------------------------------ |
| simple — fast to compute   | how to select k                      |
| converges to local minimum | sensitive to initial cluster centres |
|                            | sensitive to outliers                |
|                            | only spherical clusters              |

### EM Algorithm

_Expectation-maximization algorithm_

|        |                                                 |
| ------ | ----------------------------------------------- |
| E-Step | softly assign samples to mixture components     |
| M-Step | update the parameters of the mixture components |

#### Technical Advice\

- When implementing EM, we need to take care to avoid singularities in the estimation!\
  => Enforce minimum width for the Gaussians

- EM is very sensitive to the initialization - will converge to local minimum of $E$
  => Initialize with k-Means to get better results

- Typical procedure
  - Run k-Means M times (e.g. M = 10-100)
  - Pick the best result (lowest error J).
  - Use this result to initialize EM
    - Set $\mu_j$ to the corresponding cluster mean from k-Means.
    - Initialize $\Sigma_j$, to the sample covariance of the associated data points.

### Summary: Gaussian Mixture Models

Properties:

- Very general, can represent any (continuous) distribution.
- Once trained, very fast to evaluate.
- Can be updated online.

Problems:

- Need to apply regularization in order to avoid singularities
- EM for MoG is computationally expensive
- Need to select the number of mixture components K
  - Model selection problem

### Applications

#### Computer Vision\

- Model distributions of pixel colors.
- Each pixel is one data point in, e.g., RGB space.
- Learn a MoG to represent the class-conditional densities.
- Use the learned models to classify other pixels.

#### Background Model for Tracking\

Train background MoG for each pixel

- Model "common" appearance variation for each background pixel
- Initialization with an empty scene.
- Update the mixtures over time
  - Adapt to lighting changes, etc.

_Anything that cannot be explained by the background model is labelled as foreground_

#### Image Segmentation\

User assisted image segmentation

- User marks two regions for foreground and background.
- Learn a MoG model for the color values in each region.
- Use those models to classify all other pixels.
  - Simple segmentation procedure (building block for more complex applications)
