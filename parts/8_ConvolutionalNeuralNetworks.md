# Convolutional Neural Networks

## Neural Networks for Computer Vision

How to approach vision problems?

|                        |       |                                      |
| ---------------------- | ----- | ------------------------------------ |
| Input is 2D            | $\to$ | 2D layers of units                   |
| No pre-segmentatior    | $\to$ | Need robustness to misalignments     |
| Vision is hierarchical | $\to$ | Hierarchical multi-layered structure |
| Vision is difficult    | $\to$ | Network should be deep               |

: **Architectural considerations**

_Motivation 1:_
Visual scenes are hierarchically organized.

```
Input image -> Primitive features -> Object parts -> Object

Face Image -> Oriented Edges -> Face Parts -> Face
```

_Motivation 2:_ Biological vision is hierarchical, too.

```
Photoreceptors, retina -> Inferotemporal cortex -> V4: different textures -> V1: simple and complex cells
```

Hubel/Wiesel Architecture
~ Visual cortex consists of a hierarchy of simple, complex, and hyper-complex cells

_Motivation 3:_ Shallow architectures are inefficient at representing complex functions.

An MLP with 1 hidden layer can implement any function (universal approximator).
However, if the function is deep, a very large hidden layer may be required.

### What’s Wrong With Standard Neural Networks?

Complexity analysis:

How many parameters does this network have?
$|\theta| = 3D^2 + D$

I.e. for 32x32 image: $|\theta| = 3 * 32^4 + 32^2 \approx 3 * 10^6$

_Consequences:_

- Hard to train
- Need to initialize carefully

### Convolutional Neural Networks (CNN, ConvNet)

Neural network with specialized connectivity structure

- Stack multiple stages of feature extractors
- Higher stages compute more global, more invariant features
- Classification layer at the end

Convolutional net

- Share the same parameters across different locations
- Convolutions with learned kernels

Learn multiple filters:

- E.g. 1000x1000 image
  - 100 filters
  - 10x10 filter size
- => 10k parameters

All Neural Net activations arranged in 3 dimensions:

- Convolution layers can be stacked
- The filters of the next layer then operate on the full activation volume.
- Filters are local in (x,y), but densely connected in depth.

Stride
~ how much do you move the input across the image

---

Let’s assume the filter is an **eye detector**

_Problem_: How can we make the detection robust to the exact location of the eye?

_Solution_: By pooling (e.g., max or avg) filter responses at different spatial locations,
we gain robustness to the exact spatial location of features.

### Pooling Layers

#### Max Pooling\

|                |                                        |
| -------------- | -------------------------------------- |
| In             | 4x4                                    |
| Out            | 2x2                                    |
| Transformation | max pool with 2x2 filters and stride 2 |

: Example

Effect:

- Make the representation smaller without losing too much information
- Achieve robustness to translations

> Pooling happens independently across each slice, preserving the number of slices.

#### CNNs: Implication for Back-Propagation\

Convolutional layers

- The network parameters are in the filter masks
- Filter weights are shared between locations

=> Gradients are added for each filter location.

=> Each filter mask receives gradients from all over the image.

## CNN Architectures

### LeNet

Early convolutional architecture

- 2 Convolutional layers, 2 pooling layers
- Fully-connected N layers for classification
- Successfully used for handwritten digit recognition (MNIST)

### AlexNet

Similar framework as LeNet, but

- Bigger model (7 hidden layers, 650k units, 60M parameters)
- More data (10^6^ images instead of 10^3^)
- GPU implementation
- Better regularization and up-to-date tricks for training (Dropout)

In ILSVRC 2012 almost halved the error rate:
16.4% error (top-5) vs. 26.2% for the next best approach

### VGGNet

_Main ideas:_

- Deeper network
  - 16/19 Layers
- Stacked convolutional layers with smaller filters (+ nonlinearity)
- Detailed evaluation of all components

Results

- Improved ILSVRC top-5 error rate to 6.7%.

#### AlexNet vs. VGGNet\

|                 |        | stride |
| --------------- | ------ | ------ |
| AlexNet         | 11 x11 | 4      |
| Zeiler & Fergus | 7×7    | 2      |
| VGGNet          | 3×3    | 1      |

: Receptive fields in the first layer

Why that?

- If you stack a 3x3 on top of another 3x3 layer, you effectively get a 5x5 receptive field.
- With three 3x3 layers, the receptive field is already 7x7.
- But much fewer parameters: 3\*3^2^ = 27 instead of 7^2^ = 49.
- In addition, non-linearities in-between 3x3 layers for additional discriminativity.

### GoogLeNet

_Main ideas:_

- "Inception" module as modular component
- Learns filters at several scales within each module
- 12x fewer parameters than AlexNet -> ~5M parameters
  - Main reduction from throwing away the fully connected (FC) layers.

_Effect:_

- After last pooling layer, volume is of size `[7×7×1024]`
- Normally you would place the first 4096-D FC layer here (Many million params).
- Instead: use Average pooling in each depth slice:
  - Reduces the output to [1x1x1024].

=> Performance actually improves by 0.6% compared to when using FC layers (less overfitting?)

## Visualizing CNNs

```
A lot of images here...
```

### Inceptionism: Dreaming ConvNets

_Idea:_

- Start with a random noise image.
- Enhance the input image such as to enforce a particular response (e.g., banana).
- Combine with prior constraint that image should have similar statistics as natural images.

=> Network hallucinates characteristics of the learned class.

## Residual Networks

| No. Layers | Difficulty                          |
| ---------- | ----------------------------------- |
| 5          | easy                                |
| >10        | initialization, Batch Normalization |
| >30        | skip connections                    |
| >100       | identity skip connections           |
| >1000      | ?                                   |

: Spectrum of Depth

Deeper models are more powerful

- But training them is harder.
- Main problem: getting the gradients back to the early layers
- The deeper the network, the more effort is required for this.

---

**Is learning better networks now as simple as stacking more layers?**

General observation

- Overly deep networks have higher training error
- A general phenomenon, observed in many training sets

_Why Is That???_

Optimization difficulties - Solvers cannot find the solution when going deeper...

### Deep Residual Learning

Plain net:

`x -> |weight layer| -> relu -> |weight layer| -> relu -> H(x)`

- $H(x)$ is the desired output
- Hope the 2 weight layers fit $H(x)$

Residual net:

```
x -> |weight layer| -> relu -> |weight layer| -> relu -> H(x) = F(x) + x
   \                                                     /
    \ ------------------> identity x ------------------>/
```

- $H(x)$ is the desired output
- Hope the 2 weight layers fit $F(x)$

**$F(x)$ is a residual mapping w.r.t. identity**

- If identity were optimal, it is easy to set weights as 0
- If optimal mapping is closer to identity, it is easier to find small fluctuations
- Further advantage: direct path for the gradient to flow to the previous stages

### ResNets as ensembles of shallow networks

**The Secret Behind ResNets?**

Originally depth is good.

Now, _ensembles_ of networks are good.

Unraveling ResNets

- ResNets can be viewed as a collection of shorter paths through different subsets of the layers.
- Deleting a layer corresponds to removing only some of those paths

#### Summary\

- The effective paths in ResNets are relatively shallow

  - Effectively only 5-17 active modules

- This explains the resilience to deletion

  - Deleting any single layer only affects a subset of paths (and the shorter ones less than the longer ones).

- New interpretation of ResNets
  - ResNets work by creating an ensemble of relatively shallow paths
  - Making ResNets deeper increases the size of this ensemble
  - Excluding longer paths from training does not negatively affect the results.

## Applications of CNNs

### Object detection

Transfer Learning with CNNs:

1. Train on ImageNet

2. If small dataset: fix all weights (treat CNN as fixed feature extractor),
   retrain only the classifier

3. If you have medium sized dataset,
   “finetune” instead: use the old weights as initialization,
   train the full network or only some of the higher layers.

### Semantic segmentation

Perform pixel-wise prediction task

- Usually done using Fully Convolutional Networks (FCNs)
  - All operations formulated as convolutions
  - Advantage: can process arbitrarily sized images

Think of FCNs as performing a sliding-window classification,
producing a heatmap of output scores for each class

Encoder-Decoder Architecture

- Problem: FCN output has low resolution
- Solution: perform upsampling to get back to desired resolution
- Use skip connections to preserve higher-resolution information
