# Introduction

Machine Learning
~ Principles, methods, and algorithms for learning and prediction on the basis of past evidence

**Goal of Machine Learning:**

Machines that _learn_ to _perform_ a _task_ from _experience_.

Learning
~ most important aspect
~ We provide _data_ and _goal_ - machine figures rest out.

Learning Tools:

- statistics
- probability theory
- decision theory
- information theory
- optimization theory

## Core Questions

Task: $y = f(x;w)$

Where:

- $x$ Input
- $y$ Output
- $w$ Learned parameters

| Regression        | Classification  |
| ----------------- | --------------- |
| Continuous output | Discrete output |

## Core Problems

1. How to input data / how to interpret inputted data

2. Features

   - Invariance to irrelevant input variations
   - Selecting the "right" features is crucial
   - Encoding and use of "domain knowledge"
   - Higher-dimensional features are more discriminative.

3. Curse of Dimensionality
   - complexity increases exponentially with number of dimensions

## Core Questions

1. Measuring performance of a model

   - eg. % of correct classifications

2. Generalization performance

   - performing on test data is not enough
   - model has to perform on new data

3. What data is available?
   - Supervised vs unsupervised learning
   - mix: semi-supervised
   - reinforcement learning - with feedback

**Most often learning is an optimization problem**

I.e., maximize $y = f(x;w)$ with regard to performance.
