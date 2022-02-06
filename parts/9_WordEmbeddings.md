# Word Embeddings

## Neural Networks for Sequence Data

_Up to now:_
Simple structure `Input vector -> Processing -> Output`

_In the following:_

- we will look at sequence data
- Interesting new challenges
- Varying input/output length, need to memorize state, long-term dependencies

Currently, a hot topic:

- Early successes of NNs for text / language processing.
- Very good results for part-of-speech tagging, automatic translation, sentiment analysis, etc.
- Recently very interesting developments for video understanding,
  image+text modeling (e.g., creating image descriptions),
  and evensingle-image understanding (attention processes).

## Motivating Example

_Predicting the next word in a sequence_

### Possible solution: The trigram (n-gram) method

1. Take huge amount of text and count the frequencies of all triplets (n-tuples) of words.
2. Use those frequencies to predict the relative probabilities of words given the two previous words.

**Problems**:

1. Scalability

   - We cannot easily scale this to large N.
   - The number of possible combinations increases exponentially
   - So does the required amount of data

2. Partial Observability
   - With larger N, many counts would be zero.
   - The probability is not zero, just because the count is zero!
   - Need to back off to (N-1)-grams when the count for N-grams is too small.
   - Necessary to use elaborate techniques, such as Kneser-Ney smoothing, to compensate for uneven sampling frequencies.

### Neural Probabilistic Language Model

_Core idea:_
Learn a shared distributed encoding (word embedding) for the words in the vocabulary.

#### Word Embedding\

_Idea_

- Encode each word as a vector in a $d$-dimensional feature space.
- Typically, $V~1M$, $d \in (50, 300)$

_Learning goal_

- Determine weight matrix $W\_{V x d} that performs the embedding.
- Shared between all input words

_Input_

- Vocabulary index $x$ in 1-of-K encoding.
- For each input $x$, only one row of $W_{V x d}$ is needed.
- => $W_{V x d}$ is effectively a look-up a table.

## Popular Word Embeddings

_Open issue:_
What is the best setup for learning such an embedding from large amounts of data (billions of words)?

### word2vec

_Goal_

Make it possible to learn high-quality word embeddings from huge data sets (billions of words in training set).

_Approach_

- Define two alternative learning tasks for learning the embedding:

  - "Continuous Bag of Words" (CBOW)
  - "Skip-gram"

- Designed to require fewer parameters.

#### Continuous BOW Model\

- Remove the non-linearity from the hidden layer
- Share the projection layer for all words (their vectors are averaged)

=> Bag-of-Words model (order of the words does not matter anymore)

#### Continuous Skip-Gram Model\

- Similar structure to CBOW
- Instead of predicting the current word, predict words within a certain range of the current word.
- Give less weight to the more distant words

### Problems

With a lot of world the matrix gets too large.

=> Softmax gets expensive!

**Solution**: Hierarchical Softmax

Idea

- Organize words in binary search tree, words are at leaves
- Factorize probability of word $w_0$ as a product of node probabilities along the path.
- Learn a linear decision function $y = v\_{n(w,j)} h $ at each node to decide whether to proceed with left or right child node.

=> Decision based on output vector of hidden units directly.

## Embeddings in Vision

### Siamese networks

Similar idea to word embeddings

- Learn an embedding network that preserves (semantic) similarity between inputs
- E.g., used for patch matching

#### Discriminative Face Embeddings\

##### Triplet Loss\

1. Present the network with triplets of examples
2. Apply triplet loss to learn an embedding $f()$ that groups the positive example closer to the anchor than the negative one.

=> Used with great success in Google’s FaceNet face recognition

_Practical Issue_: How to select the triplets?

- The number of possible triplets grows cubically with the dataset size.
- Most triplets are uninformative
  -Mining hard triplets becomes crucial for learning.
- Actually want medium-hard triplets for best training efficiency

_Popular solution_: Hard triplet mining

1. Process the dataset to find hard triplets
2. Use those for learning
3. Iterate
