# Recurrent Neural Networks

_Up to now:_ Simple neural network structure: 1-to-1 mapping of inputs to outputs.

_Now_: Recurrent Neural Networks - Generalize this to arbitrary mappings.

**Example Application**:

- Part-of-Speech Tagging
- Predicting the Next Word
- Machine Translation

## Learning with RNNs

### Introduction

RNNs
~ regular NNs whose hidden units have additional forward connections over time.

- You can unroll them to create a network that extends over time.
- When you do this, keep in mind that the weights for the hidden units are shared between temporal layers.

=> With enough neurons and time, RNNs can compute anything that can be computed by your computer.

```
there is a lot of stuff that I don't understand here
```

### Problems with RNN Training

Training RNNs is very hard

- As we backpropagate through the layers,
  the magnitude of the gradient may grow or shrink exponentially
- Exploding or vanishing gradient problem!
- In an RNN trained on long sequences (e.g., 100 time steps) the gradients can easily explode or vanish.
- Even with good initial weights,
  it is very hard to detect that the current target output depends on an input from many time-steps ago.

#### Exploding / Vanishing Gradient Problem\

```
yyy something about going to infinity
and thus loosing connections in sentence
```

#### Gradient Clipping\

Trick to handle exploding gradients

- If the gradient is larger than a threshold, clip it to that threshold.

#### Handling Vanishing Gradients\

Vanishing Gradients are a harder problem

- They severely restrict the dependencies the RNN can learn.
- The problem gets more severe the deeper the network is.
- It can be very hard to diagnose that Vanishing Gradients occur
  (you just see that learning gets stuck).

Ways around the problem

- Glorot/He initialization (see Lecture 17)
- ReLU
- More complex hidden units (LSTM, GRU)

## Improved hidden units for RNNs

Target properties

- Want to achieve constant error flow through a single unit
- At the same time,
  want the unit to be able to pick up long-term connections or focus on short-term ones,
  as the problem demands.

Ideas behind LSTMs

- Take inspiration from the design of memory cells
- Keep around memories to capture long distance dependencies
- Allow error messages to flow at different strengths depending on the inputs

```
some more stuff
```
