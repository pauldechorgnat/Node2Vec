#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 13:47:39 2018

@author: paul
"""
# loading important libraries
import networkx as nx
import numpy as np
import os
import datetime
from tqdm import tqdm as prog_bar
import json

# defining a Node2Vec class
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
        self.input2hidden_weights = np.ones((self.nb_of_nodes, embeddingDimension))
        self.hidden2output_weights = np.ones((embeddingDimension, self.nb_of_nodes))
        
        # compute a probability distribution to generate negative_sampling
        probabilities = np.array([max(new_graph.in_degree[i], .001) for i in range(self.nb_of_nodes)]) 
        probabilities = np.power(probabilities, 3/4)
        probabilities /= np.sum(probabilities)
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
                break
            # computing transition probability
            probas = np.array([node['weight'] for node in outgoing_nodes.values()])
            probas = probas/sum(probas)
            # picking the next node according to transition probability
            next_node = np.random.choice(a = [node for node in outgoing_nodes.keys()], p = probas)
            # appending the next node to the walk
            walk.append(next_node)
            # shifting nodes
            current_node = next_node
        return (starting_node, walk)
        
    def create_contexts(self):
        """ creates the contexts, based on random walks
        ============================================="""
        contexts = []
        # running through the nodes 
        for node in prog_bar(self.graph.nodes):
            # for each node we want nbContexts contexts
            for i in range(self.nbContexts):
                # computing and appending the node context to the contexts list
                contexts += [self.create_random_walk(node)]
        # updating the attribute contexts
        self.contexts = contexts
        pass
    
    def create_negative_samples(self, context_node):
        negative_samples = np.random.choice(a = self.graph.nodes,
                                            size =  self.negativeRate,
                                            p = self.probabilities, 
                                            replace = False)
        return [(context_node, 1)] + [(negative_node, 0) for negative_node in negative_samples]
        
    def sigmoid(self, x, y):
        return 1/(1+np.exp(-np.sum(x*y.T)))
        
    def train(self,stepsize, epochs):
        """ trains the model using negative sampling
        =========================================
        
        INPUT :
        - stepsize : learning rate for the gradient descent
        - epochs : number of times the data is trained on"""
        
        # running through the epochs
        for epoch in range(epochs):
            print("Epoch n° : {}/{} - {}".format(epoch+1, epochs, str(datetime.datetime.now())))

            # creating contexts
            self.create_contexts()
            
            # running through contexts
            for target_node, context in prog_bar(self.contexts):
                
                h = self.input2hidden_weights[target_node,:]
                
                # creating negative samples 
                for context_node in context:
                    # generating negative samples
                    training_outputs = self.create_negative_samples(context_node)
                    # computing EH
                    EH = np.sum([(self.sigmoid(self.hidden2output_weights[:,j], h) - tj)*self.hidden2output_weights[:,j] 
                                 for j, tj in training_outputs], axis = 0)
                    
                    # updating output layer weights 
                    for j, tj in training_outputs:
                        self.hidden2output_weights[:,j] -= stepsize * (self.sigmoid(self.hidden2output_weights[:,j], h)-tj) * h.T
                    
                    # updating input layer wiegths
                    self.input2hidden_weights[target_node, :] -= stepsize * EH.T
                
        print("Training ended at ",str(datetime.datetime.now()))
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
    


