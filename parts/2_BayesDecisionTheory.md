# Bayes Decision Theory

## Probability Theory

### Probability

\begin{equation}
P(X=x) \in [0, 1]
\end{equation}

Where $X \in \{x_1, ..., x_N\}$ an Occurence of something.

---

Assuming two random variables $X \in \{x_i\}$ and $Y \in \{y_i\}$.
Consider $N$ trials and let:

$n_{ij} = count\{X = X_i \land Y = yj\}$

$c_i = count\{X = x_i\}$

$r_i = count\{Y = y_j\}$

Then we can derive _The Rules of Probability_:

|                         |                                           |
| ----------------------- | ----------------------------------------- |
| Joint Probability       | $p(X=x_i, Y= y_j) = \frac{n_{ij}}{N}$     |
| Marginal Probability    | $p(X=x_i) = \frac{c_i}{N}$                |
| Conditional Probability | $p(Y= y_j \| X=x*i) = \frac{n*{ij}}{c_i}$ |
|                         |                                           |
| Sum Rule                | $p(X) = \sum_{Y} p(X,Y)$                  |
| Product Rule            | $p(X, Y) = p(X,Y) P(X)$                   |

And _Bayes' Theorem_:

\begin{equation}
p(X|Y) = \frac{p(X|Y) P(X)}{P(X)}
\end{equation}

where: $p(X) = \sum_{Y} p(X,Y) p(Y)$

### Probability Densities

If the variable is continuous we can't just look at probability for $x$.
We have to look for the interval the $x$ is in,
using _Probability Density Function_ $p(x)$.

$p(x) = \int^{b}_{a} p(x) dx$

The probability that $x$ lies in the interval is given by $(-\infty, z)$
the _cumulative distribution function_:

$P(z) = \int_{-\infty}^{z} p(x) dx$

### Expectations

Expectation
~ average value of some function $f(x)$ under a probability distribution $p(x)$

    Discrete: $\mathbb{E}[f] = \sum_{x} p(x) f(x)$

    Continuous: $\mathbb{E}[f] = \int p(x) f(x) dx$

For finite $N$ samples we can approximate:

$\mathbb{E}[f] \backsimeq \frac{1}{N} \sum_{n=1}^{N} f(x_n)$

Conditional expectation:

$\mathbb{E}_x[f|y] = \sum_{x} p(x|y) f(x)$

### Variances and Covariances

Variance
~ measures how much variability there is in $f(x)$ around its mean value $\mathbb{E}[f(x)]$

    $var[f] = \mathbb{E}[ (f(x) -\mathbb{E}[ f(x) ] )^{2}] =\mathbb{E}[ f(x)^{2}] -\mathbb{E}[ f(x) ]^{2}$

Covariance
~ for two random variables $x$ and $y$

    $cov[x,y] = \mathbb{E}_{x,y}[ xy ] - \mathbb{E}[x] \mathbb{E}[y]$

## Bayes Decision Theory

Priors
~ a priori probabilities
~ _what can we tell about probability before seeing the data_
~ sum of all priors is $1$

Conditional probabilities
~ $p(x|C_k)$ is **likelihood** for class $C_k$

     where $x$ measures/describes certain properties of input

Posterior probabilities
~ $p(C_k | x)$
~ probability of class $C_k$ given the measurement vector $x$

    $p(C_k|x) = \frac{p(x|C_k)p(C_k)}{p(x)} = \frac{p(x|C_k)p(C_k)}{\sum_i p(x|C_i)p(C_i)}$

**Goal:**
Minimize the probability of a misclassification.

Decide for $C_k$ when:

$p(C_k|x) > p(C_j|x) \forall j \neq k$

$p(x|C_k)p(C_k) > p(x|C_j)p(C_j) \forall j \neq k$

### Classifying with Loss Functions

Motivation:
Decide if it's better to choose wrong or nothing.

In general formalized as a matrix $L_{jk}$

$L_{jk} = loss for decision C_j if C_k is correct selection$

### Minimizing the Expected Loss

Optimal solution requires knowing which class is correct - _this is unknown_.
So we **minimize the expected loss**:

$\mathbb{E}[L] = \sum_k \sum_j \int_{\mathcal{R}_j} L_{kj} p(x, C_k) dx$

This can be done by choosing the $\mathcal{R}_j$ regions such that:

$\mathbb{E}[L] = \sum_k L_{kj} p(C_k | x)$
