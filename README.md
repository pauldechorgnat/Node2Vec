# Node2Vec
Node2Vec algorithm personal Python implementation for node continuous representation

Node2Vec is an algorithm made for representing nodes in a graph as continuous vectors. 
It can be used for links predictions, nodes classifications, ... 

It is very similar to Word2Vec, an algorithm made for word representation. 

__Disclaimer__
This implementation is based on a previous work I have done to implement n-skip gram with negative sampling for word embedding. 
It is far from perfect, far from optimized. And there are still a lot of things I need to change.
There are previous implementations of Node2Vec, especially by those who invented the algorithm from [Stanford](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf). If you want to use this, it is at your own risk for now on. ;)
I'll add the other papers I used for this implementation later.

## Architecture of the algorithm

The main idea behind those algorithm is to pass a one-hot encoded vector representing a specific node through a two layers neural network (input->hidden, hidden->output) and trying to predict the probability of other nodes being the context (or neighbourhood) of the input word. 

The representation of the ith node is given by the ith line of the input->hidden weights matrix. 

## Learning 

For each node, we build a context: they are random walks from this point through the graph, where the probability transition is given by the normalized weights of the edges from the currently visited node.
Then, for each epoch and for each node, we will update weights based on the fact that the neural network should predict the nodes in the contexts.


## Optimization

To optimize the learning, we use negative sampling: for each node, for each other node in its context, we produce some negative examples, ie nodes that are not belonging to the node context.  

The way I've chosen to create negative sampling is based on the idea that very linked are less likely to be discriminant. So they can be picked up more often than less linked nodes. 

The probability of choosing a node into a negative sample is given by count(incoming edges) normalized.
