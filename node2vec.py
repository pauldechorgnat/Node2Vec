#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import random
import warnings
import json
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from gensim.models import Word2Vec


class Node2Vec():
    """wrapping the Node2Vec implementation of Aditya Grover into a class for notebook use"""
    def __init__(self,
                 dimensions = 128, walk_length = 80, num_walks = 10,
     window_size = 10, p = 1, q = 1):

        # saving parameters
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.p = p 
        self.q = q
        self.model = None
    def __str__(self):
        return 'Node2vec model'


    def load_graph_from_edgelist_file(self, input_graph, weighted = True, directed = True):
        """Reads the input network in networkx if the input is a file containing edges list"""
        # saving parameters
        self.weighted = weighted
        self.is_directed = directed
        # reading the file
        if weighted:
            G = nx.read_edgelist(input_graph, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
        else:
            G = nx.read_edgelist(input_graph, nodetype=int, create_using=nx.DiGraph())
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1

        # dealing with undirected graph
        if not directed:
            G = G.to_undirected()

        self.G = G
        pass

    def load_graph_from_edge_array(self, input_graph, weighted = True, directed = True):
        """Reads the input networkx if the input is an array of edges"""
        # saving parameters
        self.weighted = weighted
        self.is_directed = directed
        # reading the array 
        if weighted:
            G = nx.DiGraph(weighted = True)
            for u,v,weight in input_graph:
                G.add_edge(u,v, weight = weight)
        else : 
            G = DiGraph()
            for u, v in input_graph:
                G.add_edge(u,v, weight = 1)
        # dealing with indirected graph
        if not directed:
            G = g.to_undirected()

        self.G = G
        pass

    def load_graph(self, input_graph, weighted = True, directed = True):
        """Reads the input network in networkx if the input is a networkx graph:
        We just have to deal with misdefinition"""

        # saving parameters 
        self.weighted = weighted
        self.is_directed = directed

        G = nx.DiGraph(input_graph.copy(), attr = 'weight')

        # checking if the graph is weighted 
        if (not weighted) or (not  "weigth" in G.edges[list(G.edges)[0]]):
            for edge in G.edges():
                G[edge[0]][edge[1]]['weight'] = 1

        # if we want an undirected graph
        if not directed:
            G = G.to_undirected()
        '''
        else : 
            edges = set(list(G.edges(data = False)))
            for edge in edges:
                new_edge = (edge[1], edge[0])
                if new_edge not in edges:
                    G.add_edge(new_edge[0], new_edge[1], weight = 0.00001)
        '''
        self.G = G
        pass


    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        try :
            G = self.G
        except AttributeError:
            raise Exception('no graph loaded')

        is_directed = self.is_directed

        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = self.alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        pass


    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        try :
            G = self.G
        except AttributeError:
            raise Exception('no graph loaded')
            
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[self.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[self.alias_draw(alias_edges[(prev, cur)][0], 
                        alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        nodes = list(G.nodes())
        print('Simulating walks:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        
        #walks  = self.simulate_walks(self.num_walks, self.walk_length)
        self.walks = [list(map(str, walk)) for walk in walks]

        pass

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
            elif G.has_edge(dst_nbr, src):
                unnormalized_probs.append(G[dst][dst_nbr]['weight'])
            else:
                unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return self.alias_setup(normalized_probs)



    def alias_setup(self, probs):
        '''
        Compute utility lists for non-uniform sampling from discrete distributions.
        Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
        for details
        '''
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K*prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    def alias_draw(self, J, q):
        '''
        Draw sample from a non-uniform discrete distribution using alias sampling.
        '''
        K = len(J)

        kk = int(np.floor(np.random.rand()*K)) 
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]
        
    def create_model(self, workers = 8, hierarchical_softmax = 0, skipgram = 1):
        '''
        creates the model
        '''
        self.workers = workers
        self.preprocess_transition_probs()
        self.model = Word2Vec(size = self.dimensions, window = self.window_size, min_count = 0, 
            sg = skipgram, hs = hierarchical_softmax, workers = workers)
        self.simulate_walks(self.num_walks, self.walk_length)
        self.model.build_vocab(self.walks, keep_raw_vocab = True)
        

        pass

    def train(self, epochs = 10, workers = 8, verbose = False):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        if self.model is None:
            self.create_model(workers = workers)
        
        # training model 
        if verbose :        
            for i in range(epochs):
                self.model.train(self.walks, epochs = 1,  compute_loss = True, total_words = len(self.G.nodes))
                print('epoch {}/{} - loss {}'.format(i+1, epochs, self.model.get_latest_training_loss()))
        else :
            self.model.train(self.walks, epochs = epochs,  compute_loss = True, total_words = len(self.G.nodes))
        
        """self.model = Word2Vec(walks, size=self.dimensions, 
                              window=self.window_size, 
                              min_count=0, sg=1, 
                              workers=workers, iter=epochs,
                              compute_loss = True)"""
        # model.wv.save_word2vec_format(args.output)

        pass 

    def save_model(self, path):
        '''
        Saves model in a separate file
        '''
        self.model.save(path)
        pass

    def load_model(self, path):
        '''
        Loads the full Word2Vec model already trained
        '''
        self.simulate_walks(num_walks = self.num_walks, walk_length = self.walk_length)
        self.preprocess_transition_probs()
        self.model = Word2Vec.load(path)
        pass

    def save_embedding(self, path):
        '''
        saves the embedding in a JSON file
        '''
        embeddings = self.get_embedding_dictionnary()
        # json does not support np.array format
        for key in embeddings:
            embeddings[key] = embeddings[key].tolist()
        with open(path, 'w') as file:
            json.dumps(embeddings)
        pass


    def get_embedding(self, node):
        '''
        Returns the learnt representation
        '''
        return self.model.wv.get_vector(str(node))

    def get_embedding_dictionnary(self):
        '''
        Returns a dictionary whose keys are the nodes and values are the learnt representations of those nodes
        '''
        embedding_dict = {}
        for u in self.G.nodes:
            embedding_dict[u] = self.model.wv.get_vector(str(u))
        return embedding_dict
