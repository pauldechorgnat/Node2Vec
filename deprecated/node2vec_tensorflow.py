#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf
import networkx as nx
from tqdm import tqdm as prog_bar
import datetime
import numpy as np

class Node2Vec:
    
    def __init__(self, graph, embeddingDimension=200, walkLength = 3, nbContexts = 1, negativeRate=5):
        """instantiates a Node2Vec algorithm
        =================================
        INPUT :
        
        - graph : weighted graph on which the model is going to be trained. 
                  <needs to be a networkx graph>
        - embeddingDimension : dimension of continuous representation of the nodes
        - walkLength : length of the random walks used to build the contexts
        - nbContexts : number of contexts computed by node
        - negativeRate : number of negative samples for each positive sample
        """
        
        self.embeddingDimension = embeddingDimension
        self.negativeRate = negativeRate
        self.walkLength = walkLength
        self.nbContexts = nbContexts
        self.contexts = None
        self.original_graph = graph
        
        # for code simplicity, we are going to work on a copy of the graph 
        # where the names of the nodes have been replaced by an index
        
        # build a dictionnary to translate nodes name into index
        nodes = list(graph.nodes)
        self.node2index = {node:index for index, node in enumerate(nodes)}
        self.index2node = {index:node for index, node in enumerate(nodes)}
        
        # instantiating the new graph
        new_graph = nx.DiGraph()
        # adding nodes to the graph
        for node in prog_bar(nodes):
            new_graph.add_node(self.node2index[node])
        # adding edges to the graph 
        for edge in prog_bar(graph.edges(data = True)):
            u = self.node2index[edge[0]]
            v = self.node2index[edge[1]]
            w = edge[2]['weight'] if 'weight' in edge[2].keys() else 1
            if w>0:
                new_graph.add_edge(u, v, weight = w)
        
        self.graph = new_graph
        self.nb_of_nodes = len(nodes)
        
        # initializing the layers 
        self.input2hidden_weights = np.random.uniform(low = -1, high = 1, size = (self.nb_of_nodes, self.embeddingDimension))
        self.hidden2output_weights = np.random.uniform(low = -1, high = 1, size = (self.nb_of_nodes, self.embeddingDimension))
        
        # compute a probability distribution to generate negative_sampling
        probabilities = np.array([max(new_graph.in_degree[i], .001) for i in range(self.nb_of_nodes)]) 
        probabilities = np.power(probabilities, 3/4)
        probabilities = probabilities / np.sum(probabilities)
        self.probabilities = probabilities
        
        pass
    
    def create_random_walk(self, starting_node):
        """ creates a graph based random walk
        =================================
        
        INPUT :model = node2vec.Graph(graph_training, is_directed=False, p = 1, q = 1)
        - starting node : name of the node of self.graph from where the walk begins
        
        OUTPUT :
        - a tuple with shape (starting_node, [first_visited_node, second_visited_node, ...])
        """
        walk = []
        current_node = starting_node
        for i in range(self.walkLength):
            # considering the current node
            outgoing_nodes = self.graph[current_node]
            if len(outgoing_nodes) ==0:
                next_node = starting_node
            else : 
                # computing transition probability
                probas = np.array([node['weight'] for node in outgoing_nodes.values()])
                probas = probas/sum(probas)
                # picking the next node according to transition probability
                next_node = np.random.choice(a = [node for node in outgoing_nodes.keys()], p = probas)
            # appending the next node to the walk
            walk.append(next_node)
            # shifting nodes
            current_node = next_node
        return walk
    
    def create_negative_samples(self):
        """method that creates negative samples"""
        negative_samples = np.random.choice(a = self.graph.nodes,
                                            size =  (self.walkLength, self.negativeRate),
                                            p = self.probabilities, 
                                            replace = False)
        return negative_samples
    
    def create_batch_iterator(self, batch_size):
        """method that will produce an iterator among nodes"""
        # shuffling nodes
        nodes = np.random.choice(a = range(self.nb_of_nodes), size = self.nb_of_nodes, replace = False)
        # creating relevant contexts
        contexts = np.array([self.create_random_walk(starting_node = i) for i in nodes])
        # creating irrelevant samples
        negative_samples = [self.create_negative_samples() for i in nodes]

        # creating batches
        nodes = [nodes[i*batch_size: (i+1)*batch_size] for i in range(self.nb_of_nodes//batch_size)]
        contexts = [contexts[i*batch_size: (i+1)*batch_size] for i in range(self.nb_of_nodes//batch_size)]
        negative_samples = [negative_samples[i*batch_size: (i+1)*batch_size] for i in range(self.nb_of_nodes//batch_size)]
        
        
        return iter(zip(nodes, contexts, negative_samples))
     
        
        
    def sigmoid(self, x, y, axis = [2]):
        return 1/(1+tf.exp(- tf.reduce_sum(tf.multiply(x,y), axis = axis)))
        
    def train(self,stepsize, epochs, batch_size = 1, verbose = 0):
        """ trains the model using negative sampling
        =========================================
        
        INPUT :
        - stepsize : learning rate for the gradient descent
        - epochs : number of times the data is trained on"""
        
        # initializing values 
        nb_nodes = self.nb_of_nodes
        embedding_dimension = self.embeddingDimension
        context_size = self.walkLength
        negative_samples = self.negativeRate
        
        
        
        # initialize the graph with current values
        input2hidden = tf.Variable(initial_value = self.input2hidden_weights)
        hidden2output_t = tf.Variable(initial_value = self.hidden2output_weights)
        
        
        ### creating the graph 
        
        # input placeholders
        input_nodes = tf.placeholder(tf.int32, shape=[batch_size], name = 'input_nodes')
        input_context = tf.placeholder(tf.int32, shape = [batch_size, context_size], name = 'input_context_nodes')
        input_negative_samples = tf.placeholder(tf.int32, shape = [batch_size, context_size, negative_samples], 
                                                name = 'input_negative_samples_nodes')
        
        # creating higher order of input tensor to ease calculus
        target_nodes_with_context_shape = tf.reshape(tensor = tf.tile(input=input_nodes, multiples=[context_size]), 
                                                    shape = (-1, context_size))
        target_nodes_with_negative_shape = tf.reshape(tensor = tf.tile(input=input_nodes, multiples=[context_size*negative_samples]), 
                                                    shape = (-1, context_size, negative_samples))
        
        # computing the representation of the target vectors
        # useless :: h = tf.gather(indices = input_nodes, params = input2hidden)
        h_context = tf.gather(indices = target_nodes_with_context_shape, params=input2hidden)
        h_negative = tf.gather(indices = target_nodes_with_negative_shape, params = input2hidden)
        
        # computing the representations of the context/negative samples 
        vectors_context = tf.gather(indices = input_context, params = hidden2output_t)
        vectors_negative = tf.gather(indices = input_negative_samples, params = hidden2output_t)
        
        # computing the sigmoid product of the representations
        sigmoid_context = tf.reshape(
            tf.tile(self.sigmoid(h_context, vectors_context, axis = [2]), 
                multiples = [1, embedding_dimension]), 
            shape = [-1, context_size, embedding_dimension])
        sigmoid_negative = tf.reshape(
            tf.tile(self.sigmoid(h_negative, vectors_negative, axis = [3]),
                    multiples = [1, 1, embedding_dimension]), 
            shape = [-1, context_size, negative_samples, embedding_dimension])

        # computing the EH value (see paper)
        eh_context = tf.multiply(vectors_context, sigmoid_context)
        eh_negative = tf.multiply(vectors_negative, sigmoid_negative)

        eh = tf.reduce_sum(eh_context - 1, axis = [1]) + tf.reduce_sum(eh_negative, axis = [1,2])

        # computing the update of the hidden to output layer
        update_hidden2output_context = tf.multiply(sigmoid_context-1, h_context) 
        update_hidden2output_negative =  tf.reshape(tensor = tf.multiply(sigmoid_negative, h_negative), 
                                                    shape = [-1, context_size*negative_samples, embedding_dimension])
        
        
        update_hidden2output = tf.reshape(tensor = tf.concat(values = [update_hidden2output_context, update_hidden2output_negative], axis = 1),
                             shape = (-1, embedding_dimension))

        update_index_hidden2output = tf.reshape(tensor = tf.concat(
            values = [input_context, tf.reshape(input_negative_samples, (-1, context_size*negative_samples))],
            axis = 1), 
                                 shape = (-1, 1))

        hidden2output_t = tf.scatter_nd_add(ref = hidden2output_t, 
                                            indices = update_index_hidden2output, 
                                            updates = - stepsize * update_hidden2output)
        
        # computing the update of the input to hidden layer
        update_index_input2hidden = tf.reshape(input_nodes, shape = (-1, 1))

        update_input2hidden = tf.reshape(tensor = eh, shape = (-1, embedding_dimension))

        input2hidden = tf.scatter_nd_add(ref = input2hidden, 
                                         indices = update_index_input2hidden, 
                                        updates = - stepsize * update_input2hidden)
        
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        
        # running through the epochs
        for epoch in range(epochs):
            
            # printing epoch information
            if verbose : print("Epoch n° : {}/{} - {}".format(epoch+1, epochs, str(datetime.datetime.now())))
                
            # computing inputs 
            batches_iter = self.create_batch_iterator(batch_size)

            # running through batches
            for nodes, context, negative_samples in batches_iter:

                # updating

                outputs = session.run([hidden2output_t, input2hidden], 
                                     feed_dict = {input_nodes: nodes, 
                                                 input_context: context,
                                                 input_negative_samples: negative_samples})

            # saving changes
            self.input2hidden_weights = outputs[1]
            self.hidden2output_weights = outputs[0]

        print("Training ended at ",str(datetime.datetime.now()))
        pass
    def compute_similarities(self):
        """function that computes a similarity matrix using cosine similarity"""
        normalized_weigths = self.input2hidden_weights/np.linalg.norm(self.input2hidden_weights, axis = 1, keepdims=True)
        similarities = np.matmul(normalized_weigths, normalized_weigths.T)
        self.similarities = similarities
        pass
        
    def save_model(self, path):
        """
        saving model 
        ============
        
        INPUT :
        - path : path to the saving file
        """
        # transforming the data into a dictionnary
        parameters = self.__dict__.copy()
        # getting rid of the graph
        parameters.pop("graph")
        parameters.pop("original_graph")
        # changing some parameters 
        parameters['hidden2output_weights'] = parameters['hidden2output_weights'].tolist()
        parameters['input2hidden_weights'] = parameters['input2hidden_weights'].tolist()
        parameters['probabilities'] = parameters['probabilities'].tolist()
        
        with open(path, 'w', encoding = 'utf-8') as output_file:
            json.dump(parameters, fp = output_file)
        output_file.close()
        print("Model saved")
        pass
    
    @staticmethod
    def load(path, graph):
        # reading parameters
        with open(path, 'r', encoding = "utf-8") as file:
            parameters = json.load(file)
        file.close
        # instantiating without calling __init__ constructor
        node2vec = Node2Vec(graph, 
                            embeddingDimension=parameters['embeddingDimension'],
                            walkLength = parameters['walkLength'],
                            nbContexts = parameters['nbContexts'],
                            negativeRate = parameters['negativeRate'])
        
        # changing some parameters into their normal type
        input2hidden = np.array(parameters['input2hidden_weights']).reshape(parameters["nb_of_nodes"], parameters["embeddingDimension"])
        hidden2output = np.array(parameters['hidden2output_weights']).reshape(parameters["embeddingDimension"], parameters["nb_of_nodes"])
        
        node2vec.contexts = parameters['contexts']
        node2vec.input2hidden_weights = input2hidden
        node2vec.hidden2output_weights = hidden2output
        
        
        print("Model loaded")
        return node2vec     